//! Utility traits, types and functions used to define Orchard types

use std::{borrow::Borrow, marker::PhantomData};

use serde::{Deserializer, Serializer};
use serde_with::{DeserializeAs, Same, SerializeAs};

/// Abstract over how a field is owned
pub(in crate::types::orchard) trait Ownership<'a> {
    type Value<T: 'a>: Borrow<T>;
}

/// Abstract over how a slice is owned
pub(in crate::types::orchard) trait SliceOwnership<'a> {
    type Value<T: 'a>: Borrow<[T]>;
}

/// Marker type for borrowed values
pub(in crate::types::orchard) struct Borrowed<'a>(PhantomData<&'a ()>);

impl<'a, 'b> Ownership<'a> for Borrowed<'b>
where
    'a: 'b,
{
    type Value<T: 'a> = &'b T;
}

impl<'a, 'b> SliceOwnership<'a> for Borrowed<'b>
where
    'a: 'b,
{
    type Value<T: 'a> = &'b [T];
}

/// Marker type for owned values
pub(in crate::types::orchard) struct Owned;

impl<'a> Ownership<'a> for Owned {
    type Value<T: 'a> = T;
}

/// Marker type for an owned Vec
pub(in crate::types::orchard) struct OwnedVec;

impl<'a> SliceOwnership<'a> for OwnedVec {
    type Value<T: 'a> = Vec<T>;
}

/// Combinator that uses seperate encodings for deserialization and
/// serialization
pub(in crate::types::orchard) struct With<De, Ser>(PhantomData<(De, Ser)>);

impl<'de, T, De, Ser> DeserializeAs<'de, T> for With<De, Ser>
where
    De: DeserializeAs<'de, T>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        De::deserialize_as(deserializer)
    }
}

impl<T, De, Ser> SerializeAs<T> for With<De, Ser>
where
    Ser: SerializeAs<T>,
{
    fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Ser::serialize_as(source, serializer)
    }
}

/// Combinator that uses the specified encoding by reference.
pub(in crate::types::orchard) struct SerializeWithRef<As>(PhantomData<As>);

impl<'de, T, As> DeserializeAs<'de, T> for SerializeWithRef<As>
where
    As: DeserializeAs<'de, T>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        As::deserialize_as(deserializer)
    }
}

impl<'a, T, As> SerializeAs<&'a T> for SerializeWithRef<As>
where
    As: SerializeAs<T>,
{
    fn serialize_as<S>(source: &&'a T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        <&As as SerializeAs<_>>::serialize_as(source, serializer)
    }
}

/// Combinator that uses `Borrow<T>`, then applies the specified encoding.
pub(in crate::types::orchard) struct SerializeBorrow<T, As = Same>(
    PhantomData<(As, T)>,
)
where
    T: ?Sized;

impl<'de, T: ?Sized, U, As> DeserializeAs<'de, U> for SerializeBorrow<T, As>
where
    As: DeserializeAs<'de, U>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<U, D::Error>
    where
        D: Deserializer<'de>,
    {
        As::deserialize_as(deserializer)
    }
}

impl<T: ?Sized, U, As> SerializeAs<U> for SerializeBorrow<T, As>
where
    U: Borrow<T>,
    for<'a> As: SerializeAs<&'a T>,
{
    fn serialize_as<S>(source: &U, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let source: &T = source.borrow();
        As::serialize_as(&source, serializer)
    }
}

/// Combinator that converts using `TryInto` after/before deserialization/
/// serialization, respectively.
/// Similar to [`serde_with::TryFromInto`], but supports composition with
/// other combinators
pub(in crate::types::orchard) struct ComposeTryInto<T, As>(
    PhantomData<(T, As)>,
);

impl<'de, T, U, As> DeserializeAs<'de, U> for ComposeTryInto<T, As>
where
    T: TryInto<U>,
    <T as TryInto<U>>::Error: std::fmt::Display,
    As: DeserializeAs<'de, T>,
{
    fn deserialize_as<D>(deserializer: D) -> Result<U, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: T = As::deserialize_as(deserializer)?;
        T::try_into(value).map_err(<D::Error as serde::de::Error>::custom)
    }
}

impl<T, U, As> SerializeAs<T> for ComposeTryInto<U, As>
where
    for<'a> &'a T: TryInto<U>,
    for<'a> <&'a T as TryInto<U>>::Error: std::fmt::Display,
    As: SerializeAs<U>,
{
    fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let source: U = source
            .try_into()
            .map_err(<S::Error as serde::ser::Error>::custom)?;
        As::serialize_as(&source, serializer)
    }
}
