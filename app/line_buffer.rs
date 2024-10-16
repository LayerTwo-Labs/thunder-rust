use std::{
    collections::VecDeque,
    sync::{Arc, Weak},
};

use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use tracing_subscriber::fmt::MakeWriter;

const DEFAULT_MAX_CAPACITY: usize = 0x1_000;

struct LineBufferInner {
    // Max number of lines to store
    max_capacity: Option<usize>,
    // Lengths of each line within the buffer, not incluing newlines
    line_lengths: VecDeque<usize>,
    line_buffer: VecDeque<u8>,
}

impl LineBufferInner {
    fn as_str(&self) -> &str {
        // This is safe because the buffer is always checked to be valid
        // UTF-8, and the buffer is always contiguous.
        unsafe { std::str::from_utf8_unchecked(self.line_buffer.as_slices().0) }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct LineBuffer(Arc<RwLock<LineBufferInner>>);

impl LineBuffer {
    // Optionally, set the max number of lines to store
    pub fn new(max_capacity: Option<usize>) -> Self {
        let inner = LineBufferInner {
            max_capacity,
            line_lengths: VecDeque::new(),
            line_buffer: VecDeque::new(),
        };
        Self(Arc::new(RwLock::new(inner)))
    }

    pub fn as_str(&self) -> MappedRwLockReadGuard<'_, str> {
        RwLockReadGuard::map(self.0.read(), |inner| inner.as_str())
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new(Some(DEFAULT_MAX_CAPACITY))
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct LineBufferWriter(Weak<RwLock<LineBufferInner>>);

impl From<&LineBuffer> for LineBufferWriter {
    fn from(line_buffer: &LineBuffer) -> Self {
        Self(Arc::downgrade(&line_buffer.0))
    }
}

impl std::io::Write for LineBufferWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        use std::io::{Error, ErrorKind};
        if let Some(line_buffer) = self.0.upgrade() {
            // check that the buffer is valid UTF-8
            let msg = std::str::from_utf8(buf)
                .map_err(|err| Error::new(ErrorKind::InvalidData, err))?;
            let mut lines = msg.split_inclusive('\n');
            let mut line_buffer_write = line_buffer.write();
            if line_buffer_write
                .line_buffer
                .back()
                .is_some_and(|chr| *chr != b'\n')
            {
                // If the last line in the buffer is unterminated,
                // the recorded length must be incremented by length of the
                // rest of the line.
                let last_buffer_line_length =
                    line_buffer_write.line_lengths.back_mut().unwrap();
                if let Some(rest_of_line) = lines.next() {
                    *last_buffer_line_length += rest_of_line.len();
                    line_buffer_write
                        .line_buffer
                        .extend(rest_of_line.as_bytes())
                }
            }
            lines.for_each(|line| {
                line_buffer_write.line_lengths.push_back(line.len());
                line_buffer_write.line_buffer.extend(line.as_bytes());
            });
            // Remove the oldest log lines if max_capacity is set
            while line_buffer_write.max_capacity.is_some_and(|max_capacity| {
                line_buffer_write.line_lengths.len() >= max_capacity
            }) {
                let oldest_line_length =
                    line_buffer_write.line_lengths.pop_front().unwrap();
                line_buffer_write.line_buffer.drain(..oldest_line_length);
            }
            line_buffer_write.line_buffer.make_contiguous();
            drop(line_buffer_write);
            drop(line_buffer)
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl MakeWriter<'_> for LineBufferWriter {
    type Writer = Self;
    fn make_writer(&self) -> Self::Writer {
        self.clone()
    }
}
