use std::path::Path;

use acvm::{PartialWitnessGenerator, FieldElement};
use acvm::acir::circuit::{Circuit, Opcode, PublicInputs};
use acvm::acir::native_types::{Expression, Witness};
use ark_bn254::Fr;
use clap::Args;
use noirc_abi::input_parser::{Format, InputValue};
use noirc_abi::{InputMap, WitnessMap};
use noirc_driver::CompiledProgram;
use serde::Serialize;
use serde::ser::SerializeStruct;

use super::fs::{inputs::read_inputs_from_file, witness::save_witness_to_dir};
use super::NargoConfig;
use crate::{
    cli::compile_cmd::compile_circuit,
    constants::{PROVER_INPUT_FILE, TARGET_DIR},
    errors::CliError,
};

/// Executes a circuit to calculate its return value
#[derive(Debug, Clone, Args)]
pub(crate) struct ExecuteCommand {
    /// Write the execution witness to named file
    witness_name: Option<String>,

    /// Issue a warning for each unused variable instead of an error
    #[arg(short, long)]
    allow_warnings: bool,

    /// Emit debug information for the intermediate SSA IR
    #[arg(short, long)]
    show_ssa: bool,
}

pub(crate) fn run(args: ExecuteCommand, config: NargoConfig) -> Result<(), CliError> {
    let (return_value, solved_witness) =
        execute_with_path(&config.program_dir, args.show_ssa, args.allow_warnings)?;

    println!("Circuit witness successfully solved");
    if let Some(return_value) = return_value {
        println!("Circuit output: {return_value:?}");
    }
    if let Some(witness_name) = args.witness_name {
        let mut witness_dir = config.program_dir;
        witness_dir.push(TARGET_DIR);

        let witness_path = save_witness_to_dir(solved_witness, &witness_name, witness_dir)?;

        println!("Witness saved to {}", witness_path.display());
    }
    Ok(())
}

fn execute_with_path<P: AsRef<Path>>(
    program_dir: P,
    show_ssa: bool,
    allow_warnings: bool,
) -> Result<(Option<InputValue>, WitnessMap), CliError> {
    let compiled_program = compile_circuit(&program_dir, show_ssa, allow_warnings)?;

    // Parse the initial witness values from Prover.toml
    let (inputs_map, _) = read_inputs_from_file(
        &program_dir,
        PROVER_INPUT_FILE,
        Format::Toml,
        &compiled_program.abi,
    )?;

    let solved_witness = execute_program(&compiled_program, &inputs_map)?;

    let public_abi = compiled_program.abi.public_abi();
    let (_, return_value) = public_abi.decode(&solved_witness)?;

    Ok((return_value, solved_witness))
}

pub(crate) fn execute_program(
    compiled_program: &CompiledProgram,
    inputs_map: &InputMap,
) -> Result<WitnessMap, CliError> {
    let mut solved_witness = compiled_program.abi.encode(inputs_map, None)?;

    let num_witnesses = compiled_program.circuit.num_vars();
    let flattened_witnesses = (1..num_witnesses)
        .map(|wit_index| {
            // Get the value if it exists, if not then default to zero value.
            solved_witness
                .get(&Witness(wit_index))
                .map_or(FieldElement::zero(), |field| *field)
        })
        .collect();

    println!("{}", compiled_program.circuit);
    println!("{:#?}", serde_json::to_string(&RawR1CS::new(compiled_program.circuit.clone(), flattened_witnesses)).unwrap());

    let backend = crate::backends::ConcreteBackend;
    backend.solve(&mut solved_witness, compiled_program.circuit.opcodes.clone())?;

    Ok(solved_witness)
}

// AcirCircuit and AcirArithGate are R1CS-friendly structs.
//
// The difference between these structures and the ACIR structure that the compiler uses is the following:
// - The compilers ACIR struct is currently fixed to bn254
// - These structures only support arithmetic gates, while the compiler has other
// gate types. These can be added later once the backend knows how to deal with things like XOR
// or once ACIR is taught how to do convert these black box functions to Arithmetic gates.
#[derive(Clone)]
pub struct RawR1CS {
    pub gates: Vec<RawGate>,
    pub public_inputs: PublicInputs,
    pub values: Vec<Fr>,
    pub num_variables: usize,
    pub num_constraints: usize,
}

#[derive(Clone, Debug)]
pub struct RawGate {
    pub mul_terms: Vec<(Fr, Witness, Witness)>,
    pub add_terms: Vec<(Fr, Witness)>,
    pub constant_term: Fr,
}

impl RawR1CS {
    #[allow(dead_code)]
    pub fn new(acir: Circuit, values: Vec<FieldElement>) -> Self {
        let num_constraints = Self::num_constraints(&acir);
        // Currently non-arithmetic gates are not supported
        // so we extract all of the arithmetic gates only
        let gates: Vec<_> = acir
            .opcodes
            .into_iter()
            .filter(Opcode::is_arithmetic)
            .map(|opcode| RawGate::new(opcode.arithmetic().unwrap()))
            .collect();

        let values: Vec<Fr> = values.into_iter().map(from_felt).collect();

        Self {
            gates,
            values,
            num_variables: (acir.current_witness_index + 1).try_into().unwrap(),
            public_inputs: acir.public_inputs,
            num_constraints,
        }
    }

    fn num_constraints(acir: &Circuit) -> usize {
        // each multiplication term adds an extra constraint
        let mut num_opcodes = acir.opcodes.len();

        for opcode in acir.opcodes.iter() {
            match opcode {
                Opcode::Arithmetic(arith) => num_opcodes += arith.num_mul_terms() + 1, // plus one for the linear combination gate
                Opcode::Directive(_) => (),
                _ => panic!(),
            }
        }

        num_opcodes
    }
}

impl RawGate {
    #[allow(dead_code)]
    pub fn new(arithmetic_gate: Expression) -> Self {
        let converted_mul_terms: Vec<_> = arithmetic_gate
            .mul_terms
            .into_iter()
            .map(|(coefficient, multiplicand, multiplier)| {
                (from_felt(coefficient), multiplicand, multiplier)
            })
            .collect();

        let converted_linear_combinations: Vec<_> = arithmetic_gate
            .linear_combinations
            .into_iter()
            .map(|(coefficient, sum)| (from_felt(coefficient), sum))
            .collect();

        Self {
            mul_terms: converted_mul_terms,
            add_terms: converted_linear_combinations,
            constant_term: from_felt(arithmetic_gate.q_c),
        }
    }
}

fn from_felt(felt: FieldElement) -> Fr {
    felt.into_repr()
}

impl Serialize for RawR1CS {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("RawR1CS", 4)?;

        let mut serializable_values: Vec<Vec<u8>> = Vec::new();
        for value in &self.values {
            let mut serialized_value = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_uncompressed(value, &mut serialized_value)
                .map_err(|e| serde::ser::Error::custom(e.to_string()))?;
            serializable_values.push(serialized_value);
        }

        s.serialize_field("gates", &self.gates)?;
        s.serialize_field("public_inputs", &self.public_inputs)?;
        s.serialize_field("values", &serializable_values)?;
        s.serialize_field("num_variables", &self.num_variables)?;
        s.end()
    }
}

impl Serialize for RawGate {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("RawGate", 3)?;

        let mut serializable_mul_terms: Vec<(Vec<u8>, Witness, Witness)> = Vec::new();
        for (coefficient, multiplier, multiplicand) in &self.mul_terms {
            let mut serialized_coefficient = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_uncompressed(
                coefficient,
                &mut serialized_coefficient,
            )
            .map_err(|e| serde::ser::Error::custom(e.to_string()))?;
            serializable_mul_terms.push((
                serialized_coefficient,
                *multiplicand,
                *multiplier,
            ));
        }

        let mut serializable_add_terms: Vec<(Vec<u8>, Witness)> = Vec::new();
        for (coefficient, sum) in &self.add_terms {
            let mut serialized_coefficient = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_uncompressed(
                coefficient,
                &mut serialized_coefficient,
            )
            .map_err(|e| serde::ser::Error::custom(e.to_string()))?;
            serializable_add_terms.push((serialized_coefficient, *sum));
        }

        let mut serializable_constant_term = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_uncompressed(
            &self.constant_term,
            &mut serializable_constant_term,
        )
        .map_err(|e| serde::ser::Error::custom(e.to_string()))?;

        s.serialize_field("mul_terms", &serializable_add_terms)?;
        s.serialize_field("add_terms", &serializable_add_terms)?;
        s.serialize_field("constant_term", &serializable_constant_term)?;
        s.end()
    }
}
