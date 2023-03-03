use std::num::TryFromIntError;
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

    println!("UNSOLVED WITNESSES: {solved_witness:?}");
    
    let backend = crate::backends::ConcreteBackend;
    backend.solve(&mut solved_witness, compiled_program.circuit.opcodes.clone())?;
    
    println!("SOLVED WITNESSES: {solved_witness:?}");
    
    let num_witnesses = compiled_program.circuit.num_vars();
    let flattened_witnesses = (1..num_witnesses)
        .map(|wit_index| {
            // Get the value if it exists, if not then default to zero value.
            solved_witness
                .get(&Witness(wit_index))
                .map_or(FieldElement::zero(), |field| *field)
        })
        .collect();
    let r1cs = RawR1CS::new(compiled_program.circuit.clone(), flattened_witnesses);
    println!("{:?}", serde_json::to_string(&r1cs));
    
    Ok(solved_witness)
}


use ark_serialize::CanonicalSerialize;
// AcirCircuit and AcirArithGate are R1CS-friendly structs.
//
// The difference between these structures and the ACIR structure that the compiler uses is the following:
// - The compilers ACIR struct is currently fixed to bn254
// - These structures only support arithmetic gates, while the compiler has other
// gate types. These can be added later once the backend knows how to deal with things like XOR
// or once ACIR is taught how to do convert these black box functions to Arithmetic gates.
#[derive(Clone, serde::Serialize)]
pub struct RawR1CS {
    pub gates: Vec<RawGate>,
    pub public_inputs: Vec<Witness>,
    #[serde(serialize_with = "serialize_felts")]
    pub values: Vec<Fr>,
    pub num_variables: u64,
    pub num_constraints: u64,
}

#[derive(Clone, serde::Serialize)]
pub struct RawGate {
    pub mul_terms: Vec<MulTerm>,
    pub add_terms: Vec<AddTerm>,
    #[serde(serialize_with = "serialize_felt")]
    pub constant_term: Fr,
}

#[derive(Clone)]
pub struct MulTerm {
    pub coefficient: Fr,
    pub multiplicand: Witness,
    pub multiplier: Witness,
}

#[derive(Clone)]
pub struct AddTerm {
    pub coefficient: Fr,
    pub sum: Witness,
}

impl RawR1CS {
    pub fn new(
        acir: Circuit,
        values: Vec<acvm::FieldElement>,
    ) -> Self {
        let num_constraints: u64 = Self::num_constraints(&acir).try_into().unwrap();
        // Currently non-arithmetic gates are not supported
        // so we extract all of the arithmetic gates only
        let mut gates = Vec::new();
        acir.opcodes
            .into_iter()
            .filter(Opcode::is_arithmetic)
            .for_each(|opcode| {
                let expression = opcode.arithmetic().unwrap();
                gates.push(RawGate::new(expression));
            });

        let values: Vec<Fr> = values.into_iter().map(from_felt).collect();

        Self {
            gates,
            values,
            num_variables: u64::from(acir.current_witness_index) + 1,
            public_inputs: acir.public_inputs.0.into_iter().collect(),
            num_constraints,
        }
    }

    pub fn num_constraints(acir: &Circuit) -> usize {
        // each multiplication term adds an extra constraint
        let mut num_opcodes = acir.opcodes.len();

        for opcode in acir.opcodes.iter() {
            match opcode {
                Opcode::Arithmetic(arith) => num_opcodes += arith.num_mul_terms() + 1, // plus one for the linear combination gate
                Opcode::Directive(_) => (),
                _ => todo!()
            }
        }

        num_opcodes
    }
}

impl RawGate {
    pub fn new(arithmetic_gate: Expression) -> Self {
        let converted_mul_terms: Vec<MulTerm> = arithmetic_gate
            .mul_terms
            .into_iter()
            .map(|(coefficient, multiplicand, multiplier)| MulTerm {
                coefficient: from_felt(coefficient),
                multiplicand,
                multiplier,
            })
            .collect();

        let converted_linear_combinations: Vec<_> = arithmetic_gate
            .linear_combinations
            .into_iter()
            .map(|(coefficient, sum)| AddTerm {
                coefficient: from_felt(coefficient),
                sum,
            })
            .collect();

        Self {
            mul_terms: converted_mul_terms,
            add_terms: converted_linear_combinations,
            constant_term: from_felt(arithmetic_gate.q_c),
        }
    }
}

pub fn from_felt(felt: acvm::FieldElement) -> Fr {
    felt.into_repr()
}

impl Copy for MulTerm {}
impl Copy for AddTerm {}

impl std::fmt::Debug for MulTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Coefficient: {:?}", self.coefficient.0 .0)?;
        writeln!(f, "Multiplicand: {:?}", self.multiplicand.0)?;
        writeln!(f, "Multiplier: {:?}", self.multiplier.0)?;
        writeln!(f)
    }
}

impl std::fmt::Debug for AddTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Coefficient: {:?}", self.coefficient.0 .0)?;
        writeln!(f, "Sum: {:?}", self.sum.0)?;
        writeln!(f)
    }
}

impl std::fmt::Debug for RawGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.mul_terms.fmt(f)?;
        self.add_terms.fmt(f)?;
        writeln!(f, "Constant term: {:?}", self.constant_term.0 .0)?;
        writeln!(f)
    }
}

impl std::fmt::Debug for RawR1CS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.gates.fmt(f)?;
        writeln!(
            f,
            "Public Inputs: {:?}",
            self.public_inputs
                .iter()
                .map(|public_input| public_input.0)
                .collect::<Vec<_>>()
        )?;
        writeln!(
            f,
            "Values: {:?}",
            self.values
                .iter()
                .map(|value| value.0 .0)
                .collect::<Vec<_>>()
        )?;
        writeln!(f, "Number of variables: {}", self.num_variables)?;
        writeln!(f, "Number of constraints: {}", self.num_constraints)?;
        writeln!(f)
    }
}

impl Serialize for MulTerm {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut serialized_coefficient = Vec::new();
        self.coefficient
            .serialize_uncompressed(&mut serialized_coefficient)
            .map_err(serde::ser::Error::custom)?;
        // Turn little-endian to big-endian.
        serialized_coefficient.reverse();
        let encoded_coefficient = hex::encode(serialized_coefficient);

        let mut s = serializer.serialize_struct("MulTerm", 3)?;
        s.serialize_field("coefficient", &encoded_coefficient)?;
        s.serialize_field("multiplicand", &self.multiplicand)?;
        s.serialize_field("multiplier", &self.multiplier)?;
        s.end()
    }
}

impl Serialize for AddTerm {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut serialized_coefficient = Vec::new();
        self.coefficient
            .serialize_uncompressed(&mut serialized_coefficient)
            .map_err(serde::ser::Error::custom)?;
        // Turn little-endian to big-endian.
        serialized_coefficient.reverse();
        let encoded_coefficient = hex::encode(serialized_coefficient);

        let mut s = serializer.serialize_struct("AddTerm", 2)?;
        s.serialize_field("coefficient", &encoded_coefficient)?;
        s.serialize_field("sum", &self.sum)?;
        s.end()
    }
}

pub fn serialize_felt_unchecked(felt: &Fr) -> Vec<u8> {
    let mut serialized_felt = Vec::new();
    #[allow(clippy::unwrap_used)]
    felt.serialize_uncompressed(&mut serialized_felt).unwrap();
    // Turn little-endian to big-endian.
    serialized_felt.reverse();
    serialized_felt
}

pub fn serialize_felt<S>(
    felt: &Fr,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    let mut serialized_felt = Vec::new();
    felt.serialize_uncompressed(&mut serialized_felt)
        .map_err(serde::ser::Error::custom)?;
    // Turn little-endian to big-endian.
    serialized_felt.reverse();
    let encoded_coefficient = hex::encode(serialized_felt);
    serializer.serialize_str(&encoded_coefficient)
}

pub fn serialize_felts<S>(
    felts: &[Fr],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    let mut buff: Vec<u8> = Vec::new();
    let n_felts: u32 = felts.len().try_into().map_err(serde::ser::Error::custom)?;
    buff.extend_from_slice(&n_felts.to_be_bytes());
    buff.extend_from_slice(
        &felts
            .iter()
            .flat_map(serialize_felt_unchecked)
            .collect::<Vec<u8>>(),
    );
    let encoded_buff = hex::encode(buff);
    serializer.serialize_str(&encoded_buff)
}
