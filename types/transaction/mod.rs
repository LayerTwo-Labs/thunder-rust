use std::borrow::Borrow;

use bitcoin::amount::CheckedSum;
use borsh::{self, BorshSerialize};
use rustreexo::accumulator::proof::Proof as UtreexoProof;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    address::Address,
    authorization::Authorization,
    error,
    hashes::{Hash, M6id, MerkleRoot, Txid, hash_with_scratch_buffer},
    schema,
};

pub mod inputs;
pub use inputs::Inputs;
pub mod outpoint;
pub use outpoint::{OutPoint, OutPointKey};
pub mod output;
pub use output::{
    Content as OutputContent, Output, Pointed as PointedOutput,
    PointedOutputRef,
};
pub mod outputs;
pub use outputs::Outputs;

pub trait GetAddress {
    fn get_address(&self) -> Address;
}

pub trait GetValue {
    fn get_value(&self) -> bitcoin::Amount;
}

/// Reference to a tx input.
#[derive(
    Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, ToSchema,
)]
pub enum InPoint {
    /// Transaction input
    Regular {
        txid: Txid,
        // index of the spend in the inputs to spend_tx
        vin: u32,
    },
    // Created by mainchain withdrawals
    Withdrawal {
        m6id: M6id,
    },
}

#[derive(
    BorshSerialize, Clone, Debug, Default, Deserialize, Serialize, ToSchema,
)]
pub struct Transaction {
    #[schema(value_type = Vec<(OutPoint, String)>)]
    pub inputs: Inputs<(OutPoint, Hash)>,
    /// Utreexo proof for inputs
    #[borsh(skip)]
    #[schema(value_type = schema::UtreexoProof)]
    pub proof: UtreexoProof,
    pub outputs: Outputs,
}

impl Transaction {
    pub fn txid(&self) -> Txid {
        hash_with_scratch_buffer(self).into()
    }

    /// Canonical encoding as bytes. The canonical encoding is used for hashing,
    /// but other encodings may be used at eg. networking, rpc levels.
    pub fn canonical_bytes(&self) -> borsh::io::Result<Vec<u8>> {
        borsh::to_vec(&self)
    }

    /// Canonical size in bytes. The canonical encoding is used for hashing,
    /// but other encodings may be used at eg. networking, rpc levels.
    #[inline(always)]
    pub fn canonical_size(&self) -> borsh::io::Result<u64> {
        borsh::object_length(self).map(|size| size as u64)
    }

    pub(crate) fn compute_merkle_root(
        &self,
    ) -> Result<MerkleRoot, outputs::error::ComputeMerkleRoot> {
        let Self {
            inputs,
            proof: _,
            outputs,
        } = self;
        let inputs_commitment = inputs.compute_merkle_root();
        let outputs_commitment = outputs.compute_merkle_root()?;
        let res =
            hash_with_scratch_buffer(&(inputs_commitment, outputs_commitment));
        Ok(res.into())
    }
}

/// Representation of a spent output
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub struct SpentOutput {
    pub output: Output,
    pub inpoint: InPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilledTransaction {
    pub transaction: Transaction,
    pub spent_utxos: Vec<Output>,
}

impl FilledTransaction {
    pub fn get_value_in(
        &self,
    ) -> Result<bitcoin::Amount, error::AmountOverflow> {
        self.spent_utxos
            .iter()
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(error::AmountOverflow)
    }

    pub fn get_value_out(
        &self,
    ) -> Result<bitcoin::Amount, error::AmountOverflow> {
        self.transaction
            .outputs
            .iter()
            .map(GetValue::get_value)
            .checked_sum()
            .ok_or(error::AmountOverflow)
    }

    pub fn get_fee(&self) -> Result<bitcoin::Amount, error::ComputeFee> {
        let value_in = self
            .get_value_in()
            .map_err(error::ComputeFee::ValueInOverflow)?;
        let value_out = self
            .get_value_out()
            .map_err(error::ComputeFee::ValueOutOverflow)?;
        if value_in < value_out {
            Err(error::ComputeFee::Underfunded)
        } else {
            Ok(value_in - value_out)
        }
    }

    pub fn inputs(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&OutPoint, &Hash, &Output)> {
        self.transaction.inputs.iter().zip(&self.spent_utxos).map(
            |((outpoint, utxo_hash), output)| (outpoint, utxo_hash, output),
        )
    }
}

#[derive(BorshSerialize, Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct Authorized<T> {
    pub transaction: T,
    /// Authorizations are called witnesses in Bitcoin.
    pub authorizations: Vec<Authorization>,
}

pub type AuthorizedTransaction = Authorized<Transaction>;

impl<T> Borrow<T> for Authorized<T> {
    fn borrow(&self) -> &T {
        &self.transaction
    }
}

impl From<Authorized<FilledTransaction>> for AuthorizedTransaction {
    fn from(tx: Authorized<FilledTransaction>) -> Self {
        Self {
            transaction: tx.transaction.transaction,
            authorizations: tx.authorizations,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        address::Address,
        transaction::{
            FilledTransaction, GetValue, Output, OutputContent, Outputs,
            Transaction,
        },
    };

    // a withdrawal output must be funded for both its payout and its mainchain
    // fee, since both leave the treasury
    #[test]
    fn withdrawal_value_includes_main_fee() {
        let value = bitcoin::Amount::from_sat(1000);
        let main_fee = bitcoin::Amount::from_sat(300);
        let main_address = "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
            .parse::<bitcoin::Address<bitcoin::address::NetworkUnchecked>>()
            .unwrap();
        let withdrawal = Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Withdrawal {
                value,
                main_fee,
                main_address,
            },
        };
        assert_eq!(withdrawal.get_value(), value + main_fee);

        let value_output = |amount| Output {
            address: Address::ALL_ZEROS,
            content: OutputContent::Value(amount),
        };
        let withdrawal_tx = |funding| FilledTransaction {
            transaction: Transaction {
                outputs: Outputs(vec![withdrawal.clone()]),
                ..Default::default()
            },
            spent_utxos: vec![value_output(funding)],
        };

        // inputs covering only the payout are insufficient
        assert!(withdrawal_tx(value).get_fee().is_err());
        // inputs covering payout plus mainchain fee fully fund it
        assert_eq!(
            withdrawal_tx(value + main_fee).get_fee().unwrap(),
            bitcoin::Amount::ZERO
        );
    }
}
