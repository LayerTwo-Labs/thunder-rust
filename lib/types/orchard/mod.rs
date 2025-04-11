//! Orchard types

use std::{
    borrow::{Borrow, Cow},
    sync::LazyLock,
};

use anyhow::anyhow;
use borsh::BorshSerialize;
use bytemuck::{TransparentWrapper, TransparentWrapperAlloc as _};
use educe::Educe;
use incrementalmerkletree::{Position, frontier};
use nonempty::NonEmpty;
use orchard::{
    builder::BundleType,
    bundle::EffectsOnly,
    primitives::redpallas::{self, Binding, SigType},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_with::{
    Bytes, DeserializeAs, FromInto, IfIsHumanReadable, hex::Hex, serde_as,
};
use thiserror::Error;
use utoipa::ToSchema;

pub use orchard::{
    builder::{
        BuildError, BundleMetadata, InProgress, InProgressSignatures,
        OutputError, SpendError, Unauthorized, Unproven,
    },
    bundle::Authorization as BundleAuthorization,
    circuit::{ProvingKey, VerifyingKey},
    keys::{
        FullViewingKey, IncomingViewingKey, OutgoingViewingKey, Scope,
        SpendAuthorizingKey, SpendingKey,
    },
    primitives::redpallas::SpendAuth,
    tree::{MerkleHashOrchard, MerklePath},
    value::{NoteValue, OverflowError},
};

pub use crate::types::address::ShieldedAddress as Address;

pub mod shardtree_db;
mod util;

use util::{
    Borrowed, ComposeTryInto, Owned, OwnedVec, Ownership, SerializeBorrow,
    SliceOwnership, With,
};

pub use shardtree_db::{
    CreateShardTreeDbError, DbTxn as ShardTreeDbTxn, PositionWrapper,
    ShardTree, ShardTreeDb, ShardTreeError, ShardTreeStore,
    StoreError as ShardTreeStoreError,
};

/// Serde encoding for [`[u8; N]`]
type ByteArrayRepr<const N: usize> = SerializeBorrow<
    [u8; N],
    IfIsHumanReadable<With<ComposeTryInto<[u8; N], Hex>, Hex>, Bytes>,
>;

/// Serde encoding for [`Vec<u8>`]
type BytesRepr = SerializeBorrow<[u8], IfIsHumanReadable<Hex, Bytes>>;

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize, TransparentWrapper)]
#[repr(transparent)]
pub struct Signature<T: SigType>(
    #[serde_as(as = "ComposeTryInto<[u8; 64], ByteArrayRepr<{64}>>")]
    redpallas::Signature<T>,
);

impl<T> BorshSerialize for Signature<T>
where
    T: SigType,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let bytes: [u8; 64] = (&self.0).into();
        BorshSerialize::serialize(&bytes, writer)
    }
}

fn borsh_serialize_nullifier<W>(
    nullifier: &orchard::note::Nullifier,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: [u8; 32] = nullifier.to_bytes();
    BorshSerialize::serialize(&bytes, writer)
}

#[derive(
    BorshSerialize,
    Clone,
    Copy,
    Debug,
    Eq,
    Ord,
    PartialEq,
    PartialOrd,
    TransparentWrapper,
)]
#[repr(transparent)]
pub struct Nullifier(
    #[borsh(serialize_with = "borsh_serialize_nullifier")]
    orchard::note::Nullifier,
);

impl std::fmt::Display for Nullifier {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(self.0.to_bytes()).fmt(f)
    }
}

impl From<orchard::note::Nullifier> for Nullifier {
    fn from(value: orchard::note::Nullifier) -> Self {
        Self(value)
    }
}

impl<'de> Deserialize<'de> for Nullifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 32] = if deserializer.is_human_readable() {
            hex::serde::deserialize(deserializer)?
        } else {
            <[u8; 32] as Deserialize>::deserialize(deserializer)?
        };
        match orchard::note::Nullifier::from_bytes(&bytes).into_option() {
            Some(nullifier) => Ok(nullifier.into()),
            None => {
                Err(<D::Error as serde::de::Error>::custom(anyhow::anyhow!(
                    "Invalid nullifier bytes: {}",
                    hex::encode(bytes)
                )))
            }
        }
    }
}

impl Serialize for Nullifier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = self.0.to_bytes();
        if serializer.is_human_readable() {
            hex::serde::serialize(bytes, serializer)
        } else {
            <[u8; 32] as Serialize>::serialize(&bytes, serializer)
        }
    }
}

#[derive(Debug, Error)]
#[error("Error verifying signature")]
#[repr(transparent)]
pub struct SignatureVerificationError(#[from] reddsa::Error);

/// Borsh/Serde representation for [`VerificationKey`]
#[derive(Clone, Debug)]
struct VerificationKeyRepr<'a, T, O>(O::Value<redpallas::VerificationKey<T>>)
where
    T: SigType + 'a,
    O: Ownership<'a>;

impl<'a, T, O> BorshSerialize for VerificationKeyRepr<'a, T, O>
where
    T: SigType,
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let bytes = <[u8; 32] as From<_>>::from(self.0.borrow());
        BorshSerialize::serialize(&bytes, writer)
    }
}

impl<'a, T> From<&'a redpallas::VerificationKey<T>>
    for VerificationKeyRepr<'a, T, Borrowed<'a>>
where
    T: SigType,
{
    fn from(vk: &'a redpallas::VerificationKey<T>) -> Self {
        Self(vk)
    }
}

impl<'de, T> Deserialize<'de> for VerificationKeyRepr<'_, T, Owned>
where
    T: SigType,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 32] = if deserializer.is_human_readable() {
            hex::serde::deserialize(deserializer)?
        } else {
            <[u8; 32] as Deserialize>::deserialize(deserializer)?
        };
        match redpallas::VerificationKey::try_from(bytes) {
            Ok(vk) => Ok(Self(vk)),
            Err(err) => {
                let err = anyhow::Error::from(err);
                Err(<D::Error as serde::de::Error>::custom(format!("{err:#}")))
            }
        }
    }
}

impl<'a, T, O> Serialize for VerificationKeyRepr<'a, T, O>
where
    T: SigType,
    O: Ownership<'a>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes: [u8; 32] = self.0.borrow().into();
        if serializer.is_human_readable() {
            hex::serde::serialize(bytes, serializer)
        } else {
            <[u8; 32] as Serialize>::serialize(&bytes, serializer)
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[repr(transparent)]
#[serde(from = "VerificationKeyRepr<'_, T, Owned>")]
pub struct VerificationKey<T>(redpallas::VerificationKey<T>)
where
    T: SigType;

impl<T> VerificationKey<T>
where
    T: SigType,
{
    pub fn verify(
        &self,
        msg: &[u8],
        signature: &Signature<T>,
    ) -> Result<(), SignatureVerificationError> {
        self.0
            .verify(msg, Signature::peel_ref(signature))
            .map_err(SignatureVerificationError)
    }
}

impl<T> BorshSerialize for VerificationKey<T>
where
    T: SigType,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&VerificationKeyRepr::from(self), writer)
    }
}

impl<T> From<VerificationKeyRepr<'_, T, Owned>> for VerificationKey<T>
where
    T: SigType,
{
    fn from(repr: VerificationKeyRepr<'_, T, Owned>) -> Self {
        Self(repr.0)
    }
}

impl<'a, T> From<&'a VerificationKey<T>>
    for VerificationKeyRepr<'a, T, Borrowed<'a>>
where
    T: SigType,
{
    fn from(vk: &'a VerificationKey<T>) -> Self {
        Self(&vk.0)
    }
}

impl<T> Serialize for VerificationKey<T>
where
    T: SigType,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&VerificationKeyRepr::from(self), serializer)
    }
}

fn borsh_serialize_extracted_note_commitment<W>(
    cmx: &orchard::note::ExtractedNoteCommitment,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: [u8; 32] = cmx.to_bytes();
    BorshSerialize::serialize(&bytes, writer)
}

#[derive(BorshSerialize, Clone, Copy, Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct ExtractedNoteCommitment(
    #[borsh(serialize_with = "borsh_serialize_extracted_note_commitment")]
    pub  orchard::note::ExtractedNoteCommitment,
);

impl From<orchard::note::ExtractedNoteCommitment> for ExtractedNoteCommitment {
    fn from(cmx: orchard::note::ExtractedNoteCommitment) -> Self {
        Self(cmx)
    }
}

impl<'de> Deserialize<'de> for ExtractedNoteCommitment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 32] = if deserializer.is_human_readable() {
            hex::serde::deserialize(deserializer)?
        } else {
            <[u8; 32] as Deserialize>::deserialize(deserializer)?
        };
        match orchard::note::ExtractedNoteCommitment::from_bytes(&bytes)
            .into_option()
        {
            Some(cmx) => Ok(Self(cmx)),
            None => {
                let err = anyhow!(
                    "Failed to parse extracted note commitment from `{}`",
                    hex::encode(bytes)
                );
                Err(<D::Error as serde::de::Error>::custom(err))
            }
        }
    }
}

impl Serialize for ExtractedNoteCommitment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes: [u8; 32] = (&self.0).into();
        if serializer.is_human_readable() {
            hex::serde::serialize(bytes, serializer)
        } else {
            <[u8; 32] as Serialize>::serialize(&bytes, serializer)
        }
    }
}

/// Borsh/Serde representation for [`TransmittedNoteCiphertext`]
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(deserialize = "
    ByteArrayRepr<{32}>: DeserializeAs<'de, O::Value<[u8; 32]>>,
    ByteArrayRepr<{580}>: DeserializeAs<'de, O::Value<[u8; 580]>>,
    ByteArrayRepr<{80}>: DeserializeAs<'de, O::Value<[u8; 80]>>,
"))]
struct TransmittedNoteCiphertextRepr<'a, O>
where
    O: Ownership<'a>,
{
    #[serde_as(as = "ByteArrayRepr<{32}>")]
    epk_bytes: O::Value<[u8; 32]>,
    #[serde_as(as = "ByteArrayRepr<{580}>")]
    enc_ciphertext: O::Value<[u8; 580]>,
    #[serde_as(as = "ByteArrayRepr<{80}>")]
    out_ciphertext: O::Value<[u8; 80]>,
}

impl From<TransmittedNoteCiphertextRepr<'_, Owned>>
    for orchard::note::TransmittedNoteCiphertext
{
    fn from(repr: TransmittedNoteCiphertextRepr<'_, Owned>) -> Self {
        Self {
            epk_bytes: repr.epk_bytes,
            enc_ciphertext: repr.enc_ciphertext,
            out_ciphertext: repr.out_ciphertext,
        }
    }
}

impl<'a> From<&'a orchard::note::TransmittedNoteCiphertext>
    for TransmittedNoteCiphertextRepr<'a, Borrowed<'a>>
{
    fn from(
        encrypted_note: &'a orchard::note::TransmittedNoteCiphertext,
    ) -> Self {
        Self {
            epk_bytes: &encrypted_note.epk_bytes,
            enc_ciphertext: &encrypted_note.enc_ciphertext,
            out_ciphertext: &encrypted_note.out_ciphertext,
        }
    }
}

impl<'a, O> BorshSerialize for TransmittedNoteCiphertextRepr<'a, O>
where
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let Self {
            epk_bytes,
            enc_ciphertext,
            out_ciphertext,
        } = self;
        BorshSerialize::serialize(epk_bytes.borrow(), writer)?;
        BorshSerialize::serialize(enc_ciphertext.borrow(), writer)?;
        BorshSerialize::serialize(out_ciphertext.borrow(), writer)
    }
}

#[derive(Clone, Debug, Deserialize, TransparentWrapper)]
#[repr(transparent)]
#[serde(from = "TransmittedNoteCiphertextRepr<'_, Owned>")]
pub struct TransmittedNoteCiphertext(orchard::note::TransmittedNoteCiphertext);

impl TransmittedNoteCiphertext {
    fn memo_ciphertext(&self) -> [u8; 512] {
        self.0.enc_ciphertext[52..=543].try_into().unwrap()
    }

    /// Additional encrypted data after the memo field
    fn additional_ciphertext(&self) -> [u8; 20] {
        self.0.enc_ciphertext[544..=563].try_into().unwrap()
    }

    /// Authentication tag at the end of the encrypted ciphertext
    fn auth_tag_ciphertext(&self) -> [u8; 16] {
        self.0.enc_ciphertext[564..].try_into().unwrap()
    }

    /// Outgoing ciphertext
    fn out_ciphertext(&self) -> &[u8; 80] {
        &self.0.out_ciphertext
    }
}

impl BorshSerialize for TransmittedNoteCiphertext {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(
            &TransmittedNoteCiphertextRepr::from(self),
            writer,
        )
    }
}

impl From<TransmittedNoteCiphertextRepr<'_, Owned>>
    for TransmittedNoteCiphertext
{
    fn from(repr: TransmittedNoteCiphertextRepr<'_, Owned>) -> Self {
        Self(repr.into())
    }
}

impl<'a> From<&'a TransmittedNoteCiphertext>
    for TransmittedNoteCiphertextRepr<'a, Borrowed<'a>>
{
    fn from(encrypted_note: &'a TransmittedNoteCiphertext) -> Self {
        (&encrypted_note.0).into()
    }
}

impl From<orchard::note::TransmittedNoteCiphertext>
    for TransmittedNoteCiphertext
{
    fn from(encrypted_note: orchard::note::TransmittedNoteCiphertext) -> Self {
        Self(encrypted_note)
    }
}

impl Serialize for TransmittedNoteCiphertext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(
            &TransmittedNoteCiphertextRepr::from(self),
            serializer,
        )
    }
}

fn borsh_serialize_value_commitment<W>(
    cv_net: &orchard::value::ValueCommitment,
    writer: &mut W,
) -> borsh::io::Result<()>
where
    W: borsh::io::Write,
{
    let bytes: [u8; 32] = cv_net.to_bytes();
    BorshSerialize::serialize(&bytes, writer)
}

/// Borsh/Serde representation for [`ValueCommitment`]
#[derive(Clone, Debug)]
#[repr(transparent)]
struct ValueCommitmentRepr<'a, O>(O::Value<orchard::value::ValueCommitment>)
where
    O: Ownership<'a>;

impl<'a> From<&'a orchard::value::ValueCommitment>
    for ValueCommitmentRepr<'a, Borrowed<'a>>
{
    fn from(value: &'a orchard::value::ValueCommitment) -> Self {
        Self(value)
    }
}

impl<'a, O> BorshSerialize for ValueCommitmentRepr<'a, O>
where
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        borsh_serialize_value_commitment(self.0.borrow(), writer)
    }
}

impl<'de> Deserialize<'de> for ValueCommitmentRepr<'_, Owned> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 32] = if deserializer.is_human_readable() {
            hex::serde::deserialize(deserializer)?
        } else {
            <[u8; 32] as Deserialize>::deserialize(deserializer)?
        };
        match orchard::value::ValueCommitment::from_bytes(&bytes).into_option()
        {
            Some(cv_net) => Ok(Self(cv_net)),
            None => {
                let err = anyhow!(
                    "Failed to parse value commitment from `{}`",
                    hex::encode(bytes)
                );
                Err(<D::Error as serde::de::Error>::custom(err))
            }
        }
    }
}

impl<'a, O> Serialize for ValueCommitmentRepr<'a, O>
where
    O: Ownership<'a>,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes: [u8; 32] = self.0.borrow().to_bytes();
        if serializer.is_human_readable() {
            hex::serde::serialize(bytes, serializer)
        } else {
            <[u8; 32] as Serialize>::serialize(&bytes, serializer)
        }
    }
}

#[derive(BorshSerialize, Clone, Debug)]
#[repr(transparent)]
pub struct ValueCommitment(
    #[borsh(serialize_with = "borsh_serialize_value_commitment")]
    orchard::value::ValueCommitment,
);

impl From<ValueCommitmentRepr<'_, Owned>> for ValueCommitment {
    fn from(value: ValueCommitmentRepr<'_, Owned>) -> Self {
        Self(value.0)
    }
}

/// Borsh representation for [`CompactAction`]
#[derive(BorshSerialize)]
struct CompactActionRepr {
    nullifier: Nullifier,
    cmx: ExtractedNoteCommitment,
    epk_bytes: [u8; 32],
    enc_ciphertext: [u8; 52],
}

impl From<&orchard::note_encryption::CompactAction> for CompactActionRepr {
    fn from(action: &orchard::note_encryption::CompactAction) -> Self {
        use zcash_note_encryption::ShieldedOutput;
        Self {
            nullifier: action.nullifier().into(),
            cmx: action.cmx().into(),
            epk_bytes: action.ephemeral_key().0,
            enc_ciphertext: *action.enc_ciphertext(),
        }
    }
}

#[repr(transparent)]
struct CompactAction(orchard::note_encryption::CompactAction);

impl BorshSerialize for CompactAction {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&CompactActionRepr::from(self), writer)
    }
}

impl<A> From<&orchard::Action<A>> for CompactAction {
    fn from(action: &orchard::Action<A>) -> Self {
        Self(action.into())
    }
}

impl From<&CompactAction> for CompactActionRepr {
    fn from(action: &CompactAction) -> Self {
        (&action.0).into()
    }
}

/// Borsh/Serde representation for [`Action`]
#[serde_as]
#[derive(Deserialize, Educe, Serialize)]
#[educe(Debug(bound(
    VerificationKeyRepr<'a, SpendAuth, O>: std::fmt::Debug,
    O::Value<TransmittedNoteCiphertext>: std::fmt::Debug,
    ValueCommitmentRepr<'a, O>: std::fmt::Debug,
    O::Value<Auth>: std::fmt::Debug,
)))]
#[serde(bound(
    deserialize = "
        VerificationKeyRepr<'a, SpendAuth, O>: Deserialize<'de>,
        O::Value<TransmittedNoteCiphertext>: Deserialize<'de>,
        ValueCommitmentRepr<'a, O>: Deserialize<'de>,
        O::Value<Auth>: Deserialize<'de>,
    ",
    serialize = "Auth: Serialize"
))]
struct ActionRepr<'a, Auth, O>
where
    Auth: 'a,
    O: Ownership<'a>,
{
    nullifier: Nullifier,
    rk: VerificationKeyRepr<'a, SpendAuth, O>,
    cmx: ExtractedNoteCommitment,
    #[serde_as(as = "SerializeBorrow<TransmittedNoteCiphertext>")]
    encrypted_note: O::Value<TransmittedNoteCiphertext>,
    cv_net: ValueCommitmentRepr<'a, O>,
    #[serde_as(as = "SerializeBorrow<Auth>")]
    authorization: O::Value<Auth>,
}

impl<'a, Auth, O> BorshSerialize for ActionRepr<'a, Auth, O>
where
    Auth: BorshSerialize,
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let Self {
            nullifier,
            rk,
            cmx,
            encrypted_note,
            cv_net,
            authorization,
        } = self;
        BorshSerialize::serialize(nullifier, writer)?;
        BorshSerialize::serialize(rk, writer)?;
        BorshSerialize::serialize(cmx, writer)?;
        BorshSerialize::serialize(encrypted_note.borrow(), writer)?;
        BorshSerialize::serialize(cv_net, writer)?;
        BorshSerialize::serialize(authorization.borrow(), writer)
    }
}

impl<'a, Auth> From<&'a orchard::Action<Auth>>
    for ActionRepr<'a, Auth, Borrowed<'a>>
{
    fn from(action: &'a orchard::Action<Auth>) -> Self {
        Self {
            nullifier: (*action.nullifier()).into(),
            rk: action.rk().into(),
            cmx: (*action.cmx()).into(),
            encrypted_note: TransmittedNoteCiphertext::wrap_ref(
                action.encrypted_note(),
            ),
            cv_net: action.cv_net().into(),
            authorization: action.authorization(),
        }
    }
}

impl<Auth> From<ActionRepr<'_, Auth, Owned>> for orchard::Action<Auth> {
    fn from(repr: ActionRepr<'_, Auth, Owned>) -> Self {
        Self::from_parts(
            repr.nullifier.0,
            repr.rk.0,
            repr.cmx.0,
            repr.encrypted_note.0,
            ValueCommitment::from(repr.cv_net).0,
            repr.authorization,
        )
    }
}

#[derive(Clone, Debug, Deserialize, TransparentWrapper)]
#[repr(transparent)]
#[serde(
    bound = "Auth: 'de, ActionRepr<'de, Auth, Owned>: Deserialize<'de>",
    from = "ActionRepr<Auth, Owned>"
)]
pub struct Action<Auth>(orchard::Action<Auth>);

impl<Auth> Action<Auth> {
    pub fn new(
        nf: Nullifier,
        rk: VerificationKey<SpendAuth>,
        cmx: ExtractedNoteCommitment,
        encrypted_note: TransmittedNoteCiphertext,
        cv_net: ValueCommitment,
        authorization: Auth,
    ) -> Self {
        Self(orchard::Action::from_parts(
            nf.0,
            rk.0,
            cmx.0,
            encrypted_note.0,
            cv_net.0,
            authorization,
        ))
    }

    pub fn cmx(&self) -> &ExtractedNoteCommitment {
        ExtractedNoteCommitment::wrap_ref(self.0.cmx())
    }

    pub fn nullifier(&self) -> &Nullifier {
        Nullifier::wrap_ref(self.0.nullifier())
    }

    pub fn encrypted_note(&self) -> &TransmittedNoteCiphertext {
        TransmittedNoteCiphertext::wrap_ref(self.0.encrypted_note())
    }

    pub fn authorization(&self) -> &Auth {
        self.0.authorization()
    }
}

impl<Auth> From<ActionRepr<'_, Auth, Owned>> for Action<Auth> {
    fn from(repr: ActionRepr<'_, Auth, Owned>) -> Self {
        Self(repr.into())
    }
}

impl<'a, Auth> From<&'a Action<Auth>> for ActionRepr<'a, Auth, Borrowed<'a>> {
    fn from(action: &'a Action<Auth>) -> Self {
        (&action.0).into()
    }
}

impl<'a, Auth> From<&'a Action<Auth>> for CompactAction {
    fn from(action: &'a Action<Auth>) -> Self {
        (&action.0).into()
    }
}

impl<Auth> BorshSerialize for Action<Auth>
where
    Auth: BorshSerialize,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&ActionRepr::from(self), writer)
    }
}

impl<Auth> Serialize for Action<Auth>
where
    Auth: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&ActionRepr::from(self), serializer)
    }
}

impl<Auth> utoipa::PartialSchema for Action<Auth> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj = utoipa::openapi::Object::new();
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl<Auth> ToSchema for Action<Auth> {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("OrchardAction")
    }
}

#[derive(Clone, Copy, Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct BundleFlags(orchard::bundle::Flags);

impl BundleFlags {
    /// The flag set with both spends and outputs enabled.
    pub const ENABLED: Self = Self(orchard::bundle::Flags::ENABLED);

    /// The flag set with spends disabled.
    pub const SPENDS_DISABLED: Self = Self(orchard::bundle::Flags::ENABLED);

    pub fn spends_enabled(&self) -> bool {
        self.0.spends_enabled()
    }
}

impl BorshSerialize for BundleFlags {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&self.0.to_byte(), writer)
    }
}

impl<'de> Deserialize<'de> for BundleFlags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr: u8 = u8::deserialize(deserializer)?;
        match orchard::bundle::Flags::from_byte(repr) {
            Some(flags) => Ok(Self(flags)),
            None => {
                let err =
                    anyhow!("Unexpected bits set in bundle flags: {repr:x}");
                Err(<D::Error as serde::de::Error>::custom(err))
            }
        }
    }
}

impl Serialize for BundleFlags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&self.0.to_byte(), serializer)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, TransparentWrapper)]
#[repr(transparent)]
pub struct Anchor(orchard::tree::Anchor);

impl Anchor {
    pub fn empty_tree() -> Self {
        Self(orchard::tree::Anchor::empty_tree())
    }
}

impl std::fmt::Display for Anchor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        hex::encode(self.0.to_bytes()).fmt(f)
    }
}

impl BorshSerialize for Anchor {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&self.0.to_bytes(), writer)
    }
}

impl<'de> Deserialize<'de> for Anchor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: [u8; 32] = Deserialize::deserialize(deserializer)?;
        match orchard::tree::Anchor::from_bytes(bytes).into_option() {
            Some(anchor) => Ok(Self(anchor)),
            None => {
                let err = anyhow!("Invalid anchor (`{}`)", hex::encode(bytes));
                Err(<D::Error as serde::de::Error>::custom(err))
            }
        }
    }
}

impl From<MerkleHashOrchard> for Anchor {
    fn from(anchor: MerkleHashOrchard) -> Self {
        Self(anchor.into())
    }
}

impl Serialize for Anchor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&self.0.to_bytes(), serializer)
    }
}

/// Borsh/Serde representation for [`BundleProof`]
#[serde_as]
#[derive(Deserialize, Serialize)]
#[repr(transparent)]
#[serde(bound(deserialize = "BytesRepr: DeserializeAs<'de, O::Value<u8>>"))]
struct BundleProofRepr<'a, O>(#[serde_as(as = "BytesRepr")] O::Value<u8>)
where
    O: SliceOwnership<'a>;

impl<'a, O> BorshSerialize for BundleProofRepr<'a, O>
where
    O: SliceOwnership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(self.0.borrow(), writer)
    }
}

impl<'a> From<&'a orchard::Proof> for BundleProofRepr<'a, Borrowed<'a>> {
    fn from(proof: &'a orchard::Proof) -> Self {
        Self(proof.as_ref())
    }
}

impl From<BundleProofRepr<'_, OwnedVec>> for orchard::Proof {
    fn from(repr: BundleProofRepr<'_, OwnedVec>) -> Self {
        Self::new(repr.0)
    }
}

#[derive(Debug, Deserialize, TransparentWrapper)]
#[repr(transparent)]
#[serde(from = "BundleProofRepr<'_, OwnedVec>")]
pub struct BundleProof(orchard::Proof);

impl BorshSerialize for BundleProof {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&BundleProofRepr::from(self), writer)
    }
}

impl From<BundleProofRepr<'_, OwnedVec>> for BundleProof {
    fn from(repr: BundleProofRepr<'_, OwnedVec>) -> Self {
        Self(repr.into())
    }
}

impl<'a> From<&'a BundleProof> for BundleProofRepr<'a, Borrowed<'a>> {
    fn from(proof: &'a BundleProof) -> Self {
        (&proof.0).into()
    }
}

impl Serialize for BundleProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&BundleProofRepr::from(self), serializer)
    }
}

/// Borsh/Serde representation for [`Authorized`]
#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(bound(deserialize = "
    O::Value<BundleProof>: Deserialize<'de>,
    O::Value<Signature<Binding>>: Deserialize<'de>,
"))]
struct AuthorizedRepr<'a, O>
where
    O: Ownership<'a>,
{
    #[serde_as(as = "SerializeBorrow<BundleProof>")]
    proof: O::Value<BundleProof>,
    #[serde_as(as = "SerializeBorrow<Signature<Binding>>")]
    binding_signature: O::Value<Signature<Binding>>,
}

impl<'a, O> BorshSerialize for AuthorizedRepr<'a, O>
where
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let Self {
            proof,
            binding_signature,
        } = self;
        BorshSerialize::serialize(proof.borrow(), writer)?;
        BorshSerialize::serialize(binding_signature.borrow(), writer)
    }
}

impl<'a> From<&'a orchard::bundle::Authorized>
    for AuthorizedRepr<'a, Borrowed<'a>>
{
    fn from(auth: &'a orchard::bundle::Authorized) -> Self {
        Self {
            proof: BundleProof::wrap_ref(auth.proof()),
            binding_signature: Signature::wrap_ref(auth.binding_signature()),
        }
    }
}

impl From<AuthorizedRepr<'_, Owned>> for orchard::bundle::Authorized {
    fn from(repr: AuthorizedRepr<'_, Owned>) -> Self {
        Self::from_parts(repr.proof.0, repr.binding_signature.0)
    }
}

#[derive(Clone, Debug, Deserialize, TransparentWrapper)]
#[repr(transparent)]
#[serde(from = "AuthorizedRepr<'_, Owned>")]
pub struct Authorized(orchard::bundle::Authorized);

impl Authorized {
    pub fn proof(&self) -> &BundleProof {
        BundleProof::wrap_ref(self.0.proof())
    }

    pub fn binding_signature(&self) -> &Signature<Binding> {
        Signature::wrap_ref(self.0.binding_signature())
    }
}

impl BundleAuthorization for Authorized {
    type SpendAuth = Signature<SpendAuth>;
}

impl BorshSerialize for Authorized {
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&AuthorizedRepr::from(self), writer)
    }
}

impl<'a> From<&'a Authorized> for AuthorizedRepr<'a, Borrowed<'a>> {
    fn from(auth: &'a Authorized) -> Self {
        (&auth.0).into()
    }
}

impl From<AuthorizedRepr<'_, Owned>> for Authorized {
    fn from(repr: AuthorizedRepr<'_, Owned>) -> Self {
        Self(repr.into())
    }
}

impl Serialize for Authorized {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&AuthorizedRepr::from(self), serializer)
    }
}

/// Serde representation for [`Rho`]
#[serde_as]
#[derive(Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
struct RhoRepr(#[serde_as(as = "ByteArrayRepr<{32}>")] [u8; 32]);

#[derive(Clone, Copy, Debug, TransparentWrapper)]
#[repr(transparent)]
struct Rho(orchard::note::Rho);

impl<'de> Deserialize<'de> for Rho {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = RhoRepr::deserialize(deserializer)?;
        match orchard::note::Rho::from_bytes(&repr.0).into_option() {
            Some(rho) => Ok(Self(rho)),
            None => Err(<D::Error as serde::de::Error>::custom(anyhow!(
                "Invalid rho: (`{}`)",
                hex::encode(repr.0)
            ))),
        }
    }
}

impl Serialize for Rho {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = RhoRepr(self.0.to_bytes());
        Serialize::serialize(&repr, serializer)
    }
}

#[derive(Debug, Error)]
#[error(
    "Invalid rseed (`{}`) for rho (`{}`)",
    hex::encode(.rho.0.to_bytes()),
    hex::encode(.rseed),
)]
struct RandomSeedError {
    rho: Rho,
    rseed: [u8; 32],
}

/// Serde representation for [`RandomSeed`]
#[serde_as]
#[derive(Debug, Deserialize, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
struct RandomSeedRepr(#[serde_as(as = "ByteArrayRepr<{32}>")] [u8; 32]);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
struct RandomSeed(orchard::note::RandomSeed);

impl TryFrom<(&Rho, RandomSeedRepr)> for RandomSeed {
    type Error = RandomSeedError;

    fn try_from(
        (rho, repr): (&Rho, RandomSeedRepr),
    ) -> Result<Self, Self::Error> {
        match orchard::note::RandomSeed::from_bytes(repr.0, &rho.0)
            .into_option()
        {
            Some(rseed) => Ok(Self(rseed)),
            None => Err(Self::Error {
                rho: *rho,
                rseed: repr.0,
            }),
        }
    }
}

impl Serialize for RandomSeed {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = RandomSeedRepr(*self.0.as_bytes());
        Serialize::serialize(&repr, serializer)
    }
}

/// Serde representation for [`Note`]
#[derive(Debug, Deserialize, Serialize)]
struct NoteRepr {
    recipient: Address,
    value: u64,
    rho: Rho,
    rseed: RandomSeedRepr,
}

#[derive(Clone, Copy, Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct Note(orchard::Note);

impl Note {
    pub fn value(&self) -> bitcoin::Amount {
        bitcoin::Amount::from_sat(self.0.value().inner())
    }

    pub fn nullifier(&self, fvk: &FullViewingKey) -> Nullifier {
        Nullifier::wrap(self.0.nullifier(fvk))
    }
}

impl<'de> Deserialize<'de> for Note {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let NoteRepr {
            recipient,
            value,
            rho,
            rseed,
        } = NoteRepr::deserialize(deserializer)?;
        let rseed = RandomSeed::try_from((&rho, rseed)).map_err(|err| {
            <D::Error as serde::de::Error>::custom(anyhow!(err))
        })?;
        let note = orchard::Note::from_parts(
            recipient.0,
            orchard::value::NoteValue::from_raw(value),
            rho.0,
            rseed.0,
        )
        .into_option();
        match note {
            Some(rseed) => Ok(Self(rseed)),
            None => Err(<D::Error as serde::de::Error>::custom(anyhow!(
                "Invalid note"
            ))),
        }
    }
}

impl Serialize for Note {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = NoteRepr {
            recipient: Address::wrap(self.0.recipient()),
            value: self.0.value().inner(),
            rho: Rho::wrap(self.0.rho()),
            rseed: RandomSeedRepr(*self.0.rseed().as_bytes()),
        };
        Serialize::serialize(&repr, serializer)
    }
}

/// The proving key for the Orchard Action circuit
pub static PROVING_KEY: LazyLock<ProvingKey> = LazyLock::new(ProvingKey::build);

/// The verifying key for the Orchard Action circuit
pub static VERIFYING_KEY: LazyLock<VerifyingKey> =
    LazyLock::new(VerifyingKey::build);

/// Errors when trying to verify a bundle proof
#[derive(Debug, Error)]
#[error("Error verifying bundle proof")]
#[repr(transparent)]
pub struct BundleProofVerificationError(#[from] halo2_proofs::plonk::Error);

/// Safely peel the wrapper type of an [`NonEmpty<Wrapper>`]
fn peel_nonempty<T, U>(nonempty: NonEmpty<U>) -> NonEmpty<T>
where
    U: TransparentWrapper<T>,
{
    let NonEmpty { head, tail } = nonempty;
    let head = U::peel(head);
    let tail = U::peel_vec(tail);
    NonEmpty { head, tail }
}

/// Safely wrap the inner type of an [`&NonEmpty<Inner>`]
fn wrap_nonempty_ref<T, U>(nonempty: &'_ NonEmpty<T>) -> &'_ NonEmpty<U>
where
    U: TransparentWrapper<T>,
{
    // SAFETY: guaranteed to be safe by TransparentWrapper
    unsafe { std::mem::transmute(nonempty) }
}

/// Safely wrap the proof type of an [`InProgress<orchard::Proof, S>`]
fn wrap_inprogress_proof<S>(
    in_progress: InProgress<orchard::Proof, S>,
) -> InProgress<BundleProof, S>
where
    S: InProgressSignatures,
{
    // SAFETY: guaranteed to be safe by TransparentWrapper
    let res = unsafe { std::mem::transmute_copy(&in_progress) };
    std::mem::forget(in_progress);
    res
}

/// Safely peel the proof type of an [`InProgress<orchard::Proof, S>`]
fn peel_inprogress_proof<S>(
    in_progress: InProgress<BundleProof, S>,
) -> InProgress<orchard::Proof, S>
where
    S: InProgressSignatures,
{
    // SAFETY: guaranteed to be safe by TransparentWrapper
    let res = unsafe { std::mem::transmute_copy(&in_progress) };
    std::mem::forget(in_progress);
    res
}

/// Borsh/Serde representation for [`Bundle`]
#[serde_as]
#[derive(Deserialize, Educe, Serialize)]
#[educe(Debug(bound(
    O::Value<NonEmpty<Action<Auth::SpendAuth>>>: std::fmt::Debug,
    O::Value<Auth>: std::fmt::Debug,
)))]
#[serde(bound(
    serialize = "Auth: Serialize, Auth::SpendAuth: Serialize",
    deserialize = "
        O::Value<Auth>: Deserialize<'de>,
        O::Value<NonEmpty<Action<Auth::SpendAuth>>>: Deserialize<'de>
    ",
))]
struct BundleRepr<'a, Auth, O>
where
    Auth: BundleAuthorization + 'a,
    O: Ownership<'a>,
{
    #[serde_as(as = "SerializeBorrow<NonEmpty<Action<Auth::SpendAuth>>>")]
    actions: O::Value<NonEmpty<Action<Auth::SpendAuth>>>,
    flags: BundleFlags,
    #[serde(with = "bitcoin::amount::serde::as_sat")]
    balance: bitcoin::amount::SignedAmount,
    anchor: Anchor,
    #[serde_as(as = "SerializeBorrow<Auth>")]
    authorization: O::Value<Auth>,
}

impl<'a, Auth, O> BundleRepr<'a, Auth, O>
where
    Auth: BundleAuthorization + 'a,
    O: Ownership<'a>,
{
    /// Borsh serialize without including authorizations as defined in ZIP-244.
    fn borsh_serialize_without_auth<W>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()>
    where
        W: std::io::Write,
    {
        let Self {
            actions,
            flags,
            balance,
            anchor,
            authorization: _,
        } = self;
        // Compact actions
        for action in actions.borrow() {
            BorshSerialize::serialize(&CompactAction::from(action), writer)?;
        }
        // Memo items
        for action in actions.borrow() {
            let memo_ciphertext: [u8; 512] =
                action.encrypted_note().memo_ciphertext();
            BorshSerialize::serialize(&memo_ciphertext, writer)?;
            let additional_ciphertext: [u8; 20] =
                action.encrypted_note().additional_ciphertext();
            BorshSerialize::serialize(&additional_ciphertext, writer)?;
        }
        // Non-compact action items
        for action in actions.borrow() {
            let ActionRepr {
                nullifier: _,
                rk,
                cmx: _,
                encrypted_note,
                cv_net,
                authorization: _,
            } = action.into();
            BorshSerialize::serialize(&cv_net, writer)?;
            BorshSerialize::serialize(&rk, writer)?;
            let auth_tag_ciphertext: [u8; 16] =
                encrypted_note.auth_tag_ciphertext();
            BorshSerialize::serialize(&auth_tag_ciphertext, writer)?;
            let out_ciphertext: &[u8; 80] = encrypted_note.out_ciphertext();
            BorshSerialize::serialize(out_ciphertext, writer)?;
        }
        // Bundle flags
        BorshSerialize::serialize(&flags, writer)?;
        // Value balance
        BorshSerialize::serialize(&balance.to_sat(), writer)?;
        // Anchor
        BorshSerialize::serialize(&anchor, writer)?;
        Ok(())
    }
}

/// Serialization structure as defined in ZIP-244.
impl<'a, O> BorshSerialize for BundleRepr<'a, Authorized, O>
where
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let () = self.borsh_serialize_without_auth(writer)?;
        // Proof
        BorshSerialize::serialize(self.authorization.borrow().proof(), writer)?;
        // Spend Auths
        for action in self.actions.borrow() {
            BorshSerialize::serialize(action.authorization(), writer)?;
        }
        // Binding signature
        let binding_sig = self.authorization.borrow().binding_signature();
        BorshSerialize::serialize(binding_sig, writer)?;
        Ok(())
    }
}

/// Serialization structure as defined in ZIP-244.
impl<'a, O> BorshSerialize for BundleRepr<'a, EffectsOnly, O>
where
    O: Ownership<'a>,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        let () = self.borsh_serialize_without_auth(writer)?;
        Ok(())
    }
}

impl<Auth> From<BundleRepr<'_, Auth, Owned>>
    for orchard::bundle::Bundle<Auth, i64>
where
    Auth: BundleAuthorization,
{
    fn from(repr: BundleRepr<'_, Auth, Owned>) -> Self {
        Self::from_parts(
            peel_nonempty(repr.actions),
            repr.flags.0,
            repr.balance.to_sat(),
            repr.anchor.0,
            repr.authorization,
        )
    }
}

#[derive(Debug, Deserialize, Educe)]
#[educe(Clone(bound(Auth: Clone, Auth::SpendAuth: Clone)))]
#[repr(transparent)]
#[serde(
    bound = "Auth: 'de, BundleRepr<'de, Auth, Owned>: Deserialize<'de>",
    from = "BundleRepr<Auth, Owned>"
)]
pub struct Bundle<Auth>(orchard::bundle::Bundle<Auth, i64>)
where
    Auth: BundleAuthorization;

impl<Auth> Bundle<Auth>
where
    Auth: BundleAuthorization,
{
    pub fn new(
        actions: NonEmpty<Action<Auth::SpendAuth>>,
        flags: BundleFlags,
        value_balance: bitcoin::SignedAmount,
        anchor: Anchor,
        authorization: Auth,
    ) -> Self {
        Self(orchard::Bundle::from_parts(
            peel_nonempty(actions),
            flags.0,
            value_balance.to_sat(),
            anchor.0,
            authorization,
        ))
    }

    fn peel_ref<Inner>(&self) -> &Bundle<Inner>
    where
        Auth: TransparentWrapper<Inner>,
        Auth::SpendAuth: TransparentWrapper<Inner::SpendAuth>,
        Inner: BundleAuthorization,
    {
        // SAFETY: guaranteed to be safe by TransparentWrapper impls
        unsafe { std::mem::transmute(self) }
    }

    fn wrap<Wrapper>(self) -> Bundle<Wrapper>
    where
        Wrapper: TransparentWrapper<Auth>,
        Wrapper::SpendAuth: TransparentWrapper<Auth::SpendAuth>,
        Wrapper: BundleAuthorization,
    {
        Bundle(self.0.map_authorization(
            &mut (),
            |_: &mut (), _: &Auth, spend_auth: Auth::SpendAuth| {
                Wrapper::SpendAuth::wrap(spend_auth)
            },
            |_: &mut (), auth: Auth| Wrapper::wrap(auth),
        ))
    }

    pub fn actions(&self) -> &NonEmpty<Action<Auth::SpendAuth>> {
        wrap_nonempty_ref(self.0.actions())
    }

    pub fn flags(&self) -> &BundleFlags {
        BundleFlags::wrap_ref(self.0.flags())
    }

    pub fn value_balance(&self) -> bitcoin::amount::SignedAmount {
        bitcoin::amount::SignedAmount::from_sat(*self.0.value_balance())
    }

    pub fn anchor(&self) -> &Anchor {
        Anchor::wrap_ref(self.0.anchor())
    }

    pub fn authorization(&self) -> &Auth {
        self.0.authorization()
    }

    pub fn decrypt_outputs_with_keys(
        &self,
        keys: &[IncomingViewingKey],
    ) -> Vec<(usize, IncomingViewingKey, Note, Address, [u8; 512])> {
        self.0
            .decrypt_outputs_with_keys(keys)
            .into_iter()
            .map(|(idx, ivk, note, addr, memo)| {
                (idx, ivk, Note::wrap(note), Address::wrap(addr), memo)
            })
            .collect()
    }

    pub fn recover_outputs_with_ovks(
        &self,
        keys: &[OutgoingViewingKey],
    ) -> Vec<(usize, OutgoingViewingKey, Note, Address, [u8; 512])> {
        self.0
            .recover_outputs_with_ovks(keys)
            .into_iter()
            .map(|(idx, ovk, note, addr, memo)| {
                (idx, ovk, Note::wrap(note), Address::wrap(addr), memo)
            })
            .collect()
    }

    pub fn binding_validating_key(&self) -> VerificationKey<Binding> {
        VerificationKey(self.0.binding_validating_key())
    }

    /// These must be appended to the incremental note commitment merkle tree
    /// when a block is connected.
    pub fn extracted_note_commitments(
        &self,
    ) -> impl DoubleEndedIterator<Item = &ExtractedNoteCommitment> {
        self.actions().iter().map(Action::cmx)
    }

    /// Must be added to the nullifier set when a block is connected.
    pub fn nullifiers(&self) -> impl DoubleEndedIterator<Item = &Nullifier> {
        self.actions().iter().map(Action::nullifier)
    }

    /// Borsh serialize without including authorizations as defined in ZIP-244.
    pub fn borsh_serialize_without_auth<W>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()>
    where
        W: std::io::Write,
    {
        BundleRepr::from(self).borsh_serialize_without_auth(writer)
    }
}

impl<S> Bundle<InProgress<Unproven, S>>
where
    S: InProgressSignatures,
{
    pub fn create_proof<R>(
        self,
        rng: R,
    ) -> Result<Bundle<InProgress<BundleProof, S>>, BuildError>
    where
        R: rand::RngCore,
    {
        let bundle = self.0.create_proof(&PROVING_KEY, rng)?.map_authorization(
            &mut (),
            |_: &mut (),
             _: &InProgress<orchard::Proof, S>,
             spend_auth: <S as InProgressSignatures>::SpendAuth| {
                spend_auth
            },
            |&mut (), auth: InProgress<orchard::Proof, S>| {
                wrap_inprogress_proof(auth)
            },
        );
        Ok(Bundle(bundle))
    }
}

impl Bundle<InProgress<BundleProof, Unauthorized>> {
    pub fn apply_signatures<R>(
        self,
        rng: R,
        sighash: [u8; 32],
        signing_keys: &[SpendAuthorizingKey],
    ) -> Result<Bundle<Authorized>, BuildError>
    where
        R: rand::CryptoRng + rand::RngCore,
    {
        let bundle = self.0.map_authorization(
            &mut (),
            |_: &mut (), _: &InProgress<BundleProof, Unauthorized>,
                spend_auth: <Unauthorized as InProgressSignatures>::SpendAuth
            | spend_auth,
            |&mut (), auth: InProgress<BundleProof, Unauthorized>|
                peel_inprogress_proof(auth),
        );
        bundle
            .apply_signatures(rng, sighash, signing_keys)
            .map(|bundle| Bundle(bundle).wrap())
    }
}

impl Bundle<Authorized> {
    pub fn verify_proof(&self) -> Result<(), BundleProofVerificationError> {
        self.peel_ref()
            .0
            .verify_proof(&VERIFYING_KEY)
            .map_err(BundleProofVerificationError)
    }
}

impl<Auth> BorshSerialize for Bundle<Auth>
where
    Auth: BundleAuthorization,
    for<'a> BundleRepr<'a, Auth, Borrowed<'a>>: BorshSerialize,
{
    fn serialize<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        BorshSerialize::serialize(&BundleRepr::from(self), writer)
    }
}

impl<Auth> From<BundleRepr<'_, Auth, Owned>> for Bundle<Auth>
where
    Auth: BundleAuthorization,
{
    fn from(repr: BundleRepr<'_, Auth, Owned>) -> Self {
        Self(repr.into())
    }
}

impl<'a, Auth> From<&'a Bundle<Auth>> for BundleRepr<'a, Auth, Borrowed<'a>>
where
    Auth: BundleAuthorization,
{
    fn from(bundle: &'a Bundle<Auth>) -> Self {
        Self {
            actions: bundle.actions(),
            flags: *bundle.flags(),
            balance: bundle.value_balance(),
            anchor: *bundle.anchor(),
            authorization: bundle.authorization(),
        }
    }
}

impl<Auth> Serialize for Bundle<Auth>
where
    Auth: BundleAuthorization + Serialize,
    Auth::SpendAuth: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Serialize::serialize(&BundleRepr::from(self), serializer)
    }
}

impl<Auth> utoipa::PartialSchema for Bundle<Auth>
where
    Auth: BundleAuthorization,
{
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let obj = utoipa::openapi::Object::new();
        utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
    }
}

impl<Auth> utoipa::ToSchema for Bundle<Auth>
where
    Auth: BundleAuthorization,
{
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("OrchardBundle")
    }
}

pub type UnauthorizedBundle = Bundle<InProgress<Unproven, Unauthorized>>;

/// Builder for [`Bundle`]
#[derive(Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct Builder(orchard::builder::Builder);

impl Builder {
    /// If `bundle_required` is set to true, a bundle will be produced even if
    /// no spends or outputs have been added to the bundle;
    /// in such a circumstance, all of the actions in the resulting bundle will
    /// be dummies.
    pub fn new(
        flags: BundleFlags,
        bundle_required: bool,
        anchor: Anchor,
    ) -> Self {
        let bundle_type = BundleType::Transactional {
            flags: flags.0,
            bundle_required,
        };
        Self(orchard::builder::Builder::new(bundle_type, anchor.0))
    }

    pub fn add_spend(
        &mut self,
        fvk: FullViewingKey,
        note: Note,
        merkle_path: MerklePath,
    ) -> Result<(), SpendError> {
        self.0.add_spend(fvk, note.0, merkle_path)
    }

    pub fn add_output(
        &mut self,
        ovk: Option<OutgoingViewingKey>,
        recipient: Address,
        value: NoteValue,
        memo: [u8; 512],
    ) -> Result<(), OutputError> {
        self.0.add_output(ovk, recipient.0, value, memo)
    }

    pub fn value_balance(
        &self,
    ) -> Result<bitcoin::SignedAmount, OverflowError> {
        self.0.value_balance().map(bitcoin::SignedAmount::from_sat)
    }

    pub fn build<R>(
        self,
        rng: R,
    ) -> Result<Option<(UnauthorizedBundle, BundleMetadata)>, BuildError>
    where
        R: rand::RngCore,
    {
        let res = self
            .0
            .build(rng)?
            .map(|(bundle, meta)| (Bundle(bundle), meta));
        Ok(res)
    }
}

#[derive(Debug, Error)]
pub enum FrontierErrorInner {
    #[error("Position mismatch: expected {expected_ommers} ommers")]
    PositionMismatch { expected_ommers: u8 },
    #[error("Max depth exceeded (depth: {depth})")]
    MaxDepthExceeded { depth: u8 },
}

#[derive(Debug, Error)]
#[error("Frontier error")]
pub struct FrontierError(FrontierErrorInner);

impl From<incrementalmerkletree::frontier::FrontierError> for FrontierError {
    fn from(err: incrementalmerkletree::frontier::FrontierError) -> Self {
        use incrementalmerkletree::frontier;
        match err {
            frontier::FrontierError::PositionMismatch { expected_ommers } => {
                Self(FrontierErrorInner::PositionMismatch { expected_ommers })
            }
            frontier::FrontierError::MaxDepthExceeded { depth } => {
                Self(FrontierErrorInner::MaxDepthExceeded { depth })
            }
        }
    }
}

/// Serde representation for [`NonEmptyFrontier`]
#[serde_as]
#[derive(Deserialize, Serialize)]
#[serde(bound(deserialize = "
    O::Value<MerkleHashOrchard>: Deserialize<'de>,
"))]
struct NonEmptyFrontierRepr<'a, O>
where
    O: SliceOwnership<'a>,
{
    #[serde_as(as = "FromInto<u64>")]
    position: Position,
    leaf: MerkleHashOrchard,
    #[serde_as(as = "SerializeBorrow<[MerkleHashOrchard]>")]
    ommers: O::Value<MerkleHashOrchard>,
}

impl<'a> From<&'a frontier::NonEmptyFrontier<MerkleHashOrchard>>
    for NonEmptyFrontierRepr<'a, Borrowed<'a>>
{
    fn from(
        frontier: &'a frontier::NonEmptyFrontier<MerkleHashOrchard>,
    ) -> Self {
        Self {
            position: frontier.position(),
            leaf: *frontier.leaf(),
            ommers: frontier.ommers(),
        }
    }
}

/// Non-empty Orchard frontier
#[derive(Clone, Deserialize, TransparentWrapper)]
#[repr(transparent)]
#[serde(try_from = "NonEmptyFrontierRepr<'_, OwnedVec>")]
pub struct NonEmptyFrontier(frontier::NonEmptyFrontier<MerkleHashOrchard>);

impl<'a> From<&'a NonEmptyFrontier> for NonEmptyFrontierRepr<'a, Borrowed<'a>> {
    fn from(frontier: &'a NonEmptyFrontier) -> Self {
        (&frontier.0).into()
    }
}

impl TryFrom<NonEmptyFrontierRepr<'_, OwnedVec>> for NonEmptyFrontier {
    type Error = FrontierError;

    fn try_from(
        repr: NonEmptyFrontierRepr<'_, OwnedVec>,
    ) -> Result<Self, Self::Error> {
        match frontier::NonEmptyFrontier::from_parts(
            repr.position,
            repr.leaf,
            repr.ommers,
        ) {
            Ok(inner) => Ok(Self(inner)),
            Err(err) => Err(FrontierError::from(err)),
        }
    }
}

impl Serialize for NonEmptyFrontier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr = NonEmptyFrontierRepr::from(self);
        repr.serialize(serializer)
    }
}

/// Serde representation for [`Frontier`]
type FrontierRepr<'a, O> = Option<NonEmptyFrontierRepr<'a, O>>;

/// Orchard frontier
#[derive(Debug, Deserialize)]
#[repr(transparent)]
#[serde(try_from = "FrontierRepr<'_, OwnedVec>")]
pub struct Frontier(
    frontier::Frontier<
        MerkleHashOrchard,
        { orchard::NOTE_COMMITMENT_TREE_DEPTH as u8 },
    >,
);

impl Frontier {
    pub fn empty() -> Self {
        Self(frontier::Frontier::empty())
    }

    pub fn value(&self) -> Option<&NonEmptyFrontier> {
        self.0.value().map(NonEmptyFrontier::wrap_ref)
    }

    /// Consumes this wrapper and returns the underlying
    // [`Option<NonEmptyFrontier>`].
    pub fn take(self) -> Option<NonEmptyFrontier> {
        self.0.take().map(NonEmptyFrontier)
    }

    pub fn root(&self) -> MerkleHashOrchard {
        self.0.root()
    }

    /// Appends a new commitment to the frontier at the next available slot.
    /// Returns `true` if successful and `false` if the frontier would exceed
    /// the maximum allowed depth.
    pub fn append(&mut self, cmx: &ExtractedNoteCommitment) -> bool {
        self.0.append(MerkleHashOrchard::from_cmx(&cmx.0))
    }
}

impl TryFrom<NonEmptyFrontier> for Frontier {
    type Error = FrontierError;

    fn try_from(non_empty: NonEmptyFrontier) -> Result<Self, Self::Error> {
        match non_empty.0.try_into() {
            Ok(frontier) => Ok(Self(frontier)),
            Err(err) => Err(err.into()),
        }
    }
}

impl TryFrom<FrontierRepr<'_, OwnedVec>> for Frontier {
    type Error = FrontierError;

    fn try_from(repr: FrontierRepr<'_, OwnedVec>) -> Result<Self, Self::Error> {
        let Some(repr) = repr else {
            return Ok(Self(frontier::Frontier::empty()));
        };
        let non_empty = NonEmptyFrontier::try_from(repr)?;
        non_empty.try_into()
    }
}

impl Serialize for Frontier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let repr: FrontierRepr<Borrowed<'_>> =
            self.0.value().map(NonEmptyFrontierRepr::from);
        repr.serialize(serializer)
    }
}
