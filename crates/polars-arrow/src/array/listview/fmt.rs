use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::write_vec;
use super::ListViewArrayGeneric;
use crate::array::listview::{ViewType, ListViewArrayGeneric};
use crate::array::Array;

pub fn write_value<'a, T: ViewType + ?Sized, W: Write>(
    array: &'a ListViewArrayGeneric<T>,
    index: usize,
    f: &mut W,
) -> Result
where
    &'a T: Debug,
{
    let bytes = array.value(index).to_bytes();
    let writer = |f: &mut W, index| write!(f, "{}", bytes[index]);

    write_vec(f, writer, None, bytes.len(), "None", false)
}

impl Debug for ListViewArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, f);
        write!(f, "ListViewArray")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}


