//! Utility types and functions for this crate

/// Borsh encoding and decoding
pub(crate) mod borsh {
    /// Borsh encoding
    pub mod serialize {
        use borsh::BorshSerialize;

        use crate::{
            UtreexoNodeHash,
            authorization::{Signature, VerifyingKey},
        };

        pub fn bitcoin_block_hash<W>(
            block_hash: &bitcoin::BlockHash,
            writer: &mut W,
        ) -> borsh::io::Result<()>
        where
            W: borsh::io::Write,
        {
            let bytes: &[u8; 32] = block_hash.as_ref();
            BorshSerialize::serialize(bytes, writer)
        }

        pub fn signature<W>(
            sig: &Signature,
            writer: &mut W,
        ) -> borsh::io::Result<()>
        where
            W: borsh::io::Write,
        {
            borsh::BorshSerialize::serialize(&sig.to_bytes(), writer)
        }

        pub fn utreexo_node_hash<W>(
            node_hash: &UtreexoNodeHash,
            writer: &mut W,
        ) -> borsh::io::Result<()>
        where
            W: borsh::io::Write,
        {
            let bytes: &[u8; 32] = node_hash;
            BorshSerialize::serialize(bytes, writer)
        }

        pub fn utreexo_roots<W>(
            roots: &[UtreexoNodeHash],
            writer: &mut W,
        ) -> borsh::io::Result<()>
        where
            W: borsh::io::Write,
        {
            #[derive(BorshSerialize)]
            #[repr(transparent)]
            struct SerializeUtreexoNodeHash<'a>(
                #[borsh(serialize_with = "utreexo_node_hash")]
                &'a UtreexoNodeHash,
            );
            let roots: Vec<SerializeUtreexoNodeHash> =
                roots.iter().map(SerializeUtreexoNodeHash).collect();
            BorshSerialize::serialize(&roots, writer)
        }

        pub fn verifying_key<W>(
            vk: &VerifyingKey,
            writer: &mut W,
        ) -> borsh::io::Result<()>
        where
            W: borsh::io::Write,
        {
            borsh::BorshSerialize::serialize(&vk.to_bytes(), writer)
        }
    }
}

/// Serde adapters
pub(crate) mod serde {
    /// (de)serialize as hex strings for human-readable forms like json,
    /// and default serialization for non human-readable formats like bincode
    pub mod hexstr_human_readable {
        use const_hex::{FromHex, ToHexExt};
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        pub fn serialize<S, T>(
            data: T,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
            T: Serialize + ToHexExt,
        {
            if serializer.is_human_readable() {
                data.encode_hex().serialize(serializer)
            } else {
                data.serialize(serializer)
            }
        }

        pub fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
        where
            D: Deserializer<'de>,
            T: Deserialize<'de> + FromHex,
            <T as FromHex>::Error: std::fmt::Display,
        {
            if deserializer.is_human_readable() {
                const_hex::serde::deserialize(deserializer)
            } else {
                T::deserialize(deserializer)
            }
        }
    }
}
