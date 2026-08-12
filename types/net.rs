use crate::THIS_SIDECHAIN;

pub const DEFAULT_PORT: u16 = 4000 + THIS_SIDECHAIN as u16;

pub mod peer {
    use std::{
        borrow::ToOwned,
        fmt::Display,
        net::{IpAddr, SocketAddr},
        str::FromStr,
    };

    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use strum::Display;
    use utoipa::ToSchema;

    use crate::schema;

    pub type ParseAddressError = crate::error::ParsePeerAddress;

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct Address<S = String> {
        pub host: url::Host<S>,
        pub port: u16,
    }

    impl<S> Address<S> {
        pub fn as_ref(&self) -> Address<&S> {
            let Self { host, port } = self;
            let host = match host {
                url::Host::Domain(domain) => url::Host::Domain(domain),
                url::Host::Ipv4(v4) => url::Host::Ipv4(*v4),
                url::Host::Ipv6(v6) => url::Host::Ipv6(*v6),
            };
            Address { host, port: *port }
        }

        pub fn map_domain<F, T>(self, f: F) -> Address<T>
        where
            F: FnOnce(S) -> T,
        {
            let Self { host, port } = self;
            let host = match host {
                url::Host::Domain(domain) => url::Host::Domain(f(domain)),
                url::Host::Ipv4(v4) => url::Host::Ipv4(v4),
                url::Host::Ipv6(v6) => url::Host::Ipv6(v6),
            };
            Address { host, port }
        }
    }

    impl<S> Address<&S>
    where
        S: ?Sized,
    {
        pub fn to_owned(&self) -> Address<<S as ToOwned>::Owned>
        where
            S: ToOwned,
        {
            self.as_ref()
                .map_domain(|domain| <S as ToOwned>::to_owned(domain))
        }
    }

    impl<'de> Deserialize<'de> for Address {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            if deserializer.is_human_readable() {
                let s: &'_ str = Deserialize::deserialize(deserializer)?;
                <Self as FromStr>::from_str(s).map_err(serde::de::Error::custom)
            } else {
                // Representation for serde
                #[derive(Deserialize)]
                struct AddressRepr {
                    host: url::Host,
                    port: u16,
                }
                let AddressRepr { host, port } =
                    Deserialize::deserialize(deserializer)?;
                Ok(Self { host, port })
            }
        }
    }

    impl<S> Display for Address<S>
    where
        url::Host<S>: Display,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let Self { host, port } = self;
            write!(f, "{host}:{port}")
        }
    }

    impl<S, T> From<T> for Address<S>
    where
        SocketAddr: From<T>,
    {
        fn from(value: T) -> Self {
            match SocketAddr::from(value) {
                SocketAddr::V4(v4) => Self {
                    host: url::Host::Ipv4(*v4.ip()),
                    port: v4.port(),
                },
                SocketAddr::V6(v6) => Self {
                    host: url::Host::Ipv6(*v6.ip()),
                    port: v6.port(),
                },
            }
        }
    }

    /// Parses `host:port`. IPv6 literals must be bracketed, as in `[::1]:4009`, so
    /// their colons are not read as the port separator.
    impl FromStr for Address {
        type Err = ParseAddressError;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            let (host_str, port) = match s.rsplit_once(':') {
                Some((host_str, port)) => {
                    let port: u16 = port
                        .parse()
                        .map_err(|_| url::ParseError::InvalidPort)?;
                    (host_str, port)
                }
                None => (s, crate::net::DEFAULT_PORT),
            };
            let host = url::Host::parse(host_str)?;
            Ok(Self { host, port })
        }
    }

    impl<T> Serialize for Address<T>
    where
        url::Host<T>: std::fmt::Display + Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            if serializer.is_human_readable() {
                serializer.serialize_str(&self.to_string())
            } else {
                // Representation for serde
                #[derive(Serialize)]
                #[serde(bound(serialize = "url::Host<T>: Serialize"))]
                struct AddressRepr<'a, T> {
                    host: &'a url::Host<T>,
                    port: u16,
                }
                let Self { host, port } = self;
                let repr = AddressRepr { host, port: *port };
                repr.serialize(serializer)
            }
        }
    }

    impl<S> utoipa::PartialSchema for Address<S> {
        fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
            let obj = utoipa::openapi::Object::with_type(
                utoipa::openapi::Type::String,
            );
            utoipa::openapi::RefOr::T(utoipa::openapi::Schema::Object(obj))
        }
    }

    impl<S> utoipa::ToSchema for Address<S> {
        fn name() -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("PeerAddress")
        }
    }

    /// A peer address that has been resolved to one or more IP address
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum ResolvedAddress<S = String> {
        Domain {
            port: u16,
            /// Addresses, in reverse order such that the last element is the
            /// first resolved address.
            addrs: nonempty::NonEmpty<IpAddr>,
            domain: S,
        },
        /// Resolution is not required for static addrs
        Static(SocketAddr),
    }

    impl<S> ResolvedAddress<S> {
        pub fn host(&self) -> url::Host<&S> {
            match self {
                Self::Domain { domain, .. } => url::Host::Domain(domain),
                Self::Static(SocketAddr::V4(v4)) => url::Host::Ipv4(*v4.ip()),
                Self::Static(SocketAddr::V6(v6)) => url::Host::Ipv6(*v6.ip()),
            }
        }

        pub fn port(&self) -> u16 {
            match self {
                Self::Domain { port, .. } => *port,
                Self::Static(addr) => addr.port(),
            }
        }

        pub fn as_peer_address(&self) -> Address<&S> {
            Address {
                host: self.host(),
                port: self.port(),
            }
        }

        /// first resolved IP addr
        pub fn first_ip_addr(&self) -> IpAddr {
            match self {
                Self::Domain { addrs, .. } => *addrs.last(),
                Self::Static(addr) => addr.ip(),
            }
        }

        pub fn pop_first_ip_addr(self) -> (IpAddr, Option<Self>) {
            match self {
                Self::Domain {
                    domain,
                    mut addrs,
                    port,
                } => {
                    if let Some(addr) = addrs.pop() {
                        (
                            addr,
                            Some(Self::Domain {
                                port,
                                addrs,
                                domain,
                            }),
                        )
                    } else {
                        (addrs.head, None)
                    }
                }
                Self::Static(addr) => (addr.ip(), None),
            }
        }

        pub fn ip_addrs(&self) -> impl Iterator<Item = IpAddr> {
            match self {
                Self::Domain { addrs, .. } => {
                    Box::new(addrs.iter().rev().cloned())
                        as Box<dyn Iterator<Item = IpAddr>>
                }
                Self::Static(addr) => Box::new(std::iter::once(addr.ip())),
            }
        }
    }

    impl<S, T> From<T> for ResolvedAddress<S>
    where
        SocketAddr: From<T>,
    {
        fn from(value: T) -> Self {
            Self::Static(SocketAddr::from(value))
        }
    }

    #[derive(
        Clone, Copy, Deserialize, Display, Eq, PartialEq, Serialize, ToSchema,
    )]
    #[schema(as = PeerConnectionStatus)]
    pub enum ConnectionStatus {
        /// We're still in the process of initializing the peer connection
        Connecting,
        /// The connection is successfully established
        Connected,
    }

    /// RPC output representation for peer + state
    #[derive(Clone, Deserialize, Serialize, ToSchema)]
    pub struct Peer {
        #[schema(value_type = schema::SocketAddr)]
        pub address: SocketAddr,
        pub status: ConnectionStatus,
    }
}
pub use peer::{
    Address as PeerAddress, ConnectionStatus as PeerConnectionStatus, Peer,
    ResolvedAddress as ResolvedPeerAddress,
};
