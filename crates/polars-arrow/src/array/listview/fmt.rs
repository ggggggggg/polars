use std::fmt::{Debug, Formatter, Result, Write};

use super::super::fmt::{get_display, write_vec};
use super::ListViewArray;
use crate::offset::Offset;

pub fn write_value<O: Offset, W: Write>(
    array: &ListViewArray<O>,
    index: usize,
    null: &'static str,
    f: &mut W,
) -> Result {
    let values = array.value(index);
    let writer = |f: &mut W, index| get_display(values.as_ref(), null)(f, index);
    write_vec(f, writer, None, values.len(), null, false)
}

impl<O: Offset> Debug for ListViewArray<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let writer = |f: &mut Formatter, index| write_value(self, index, "None", f);

        let head = if O::IS_LARGE {
            "LargeListViewArray"
        } else {
            "ListViewArray"
        };
        write!(f, "{head}")?;
        write_vec(f, writer, self.validity(), self.len(), "None", false)
    }
}
