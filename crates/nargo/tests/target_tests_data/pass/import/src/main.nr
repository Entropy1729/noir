mod import;
use crate::import::hello;

fn main(x : Field, y : Field) {
    let _k = dep::std::hash::pedersen([x]);
    let _l = hello(x);

    constrain x != import::hello(y);
}
