use dep::std;

fn len_plus_1<T>(array: [T]) -> Field {
    array.len() + 1
}

fn add_lens<T>(a: [T], b: [Field]) -> Field {
    a.len() + b.len()
}

fn nested_call(b: [Field]) -> Field {
    len_plus_1(b)
}

fn main(len3: [u8; 3], len4: [Field; 4]) {
    constrain len_plus_1(len3) == 4;
    constrain len_plus_1(len4) == 5;
    constrain add_lens(len3, len4) == 7;
    constrain nested_call(len4) == 5;

    // std::array::len returns a comptime value
    constrain len4[len3.len()] == 4;
    
    // test for std::array::sort
    let mut unsorted = len3;
    unsorted[0] = len3[1];
    unsorted[1] = len3[0];
    constrain unsorted[0] > unsorted[1];
    let sorted = unsorted.sort();
    constrain sorted[0] < sorted[1];
}
