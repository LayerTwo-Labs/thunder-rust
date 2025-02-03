use nonempty::NonEmpty;
use serde::{Deserialize, Serialize};

/// Data of type `T` paired with block height at which it was last updated
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HeightStamped<T> {
    pub value: T,
    pub height: u32,
}

/// Wrapper struct for fields that support rollbacks
#[derive(Clone, Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct RollBack<T>(NonEmpty<HeightStamped<T>>);

impl<T> RollBack<T> {
    pub fn new(value: T, height: u32) -> Self {
        let height_stamped = HeightStamped { value, height };
        Self(NonEmpty::new(height_stamped))
    }

    /// Pop the most recent value
    pub fn pop(mut self) -> (Option<Self>, HeightStamped<T>) {
        if let Some(value) = self.0.pop() {
            (Some(self), value)
        } else {
            (None, self.0.head)
        }
    }

    /// Attempt to push a value as the new most recent.
    /// Returns the value if the operation fails.
    pub fn push(&mut self, value: T, height: u32) -> Result<(), T> {
        if self.0.last().height > height {
            return Err(value);
        }
        let height_stamped = HeightStamped { value, height };
        self.0.push(height_stamped);
        Ok(())
    }

    /// Returns the earliest value
    pub fn earliest(&self) -> &HeightStamped<T> {
        self.0.first()
    }

    /// Returns the most recent value
    pub fn latest(&self) -> &HeightStamped<T> {
        self.0.last()
    }
}
