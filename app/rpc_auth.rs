//! Cookie-file authentication for the private / wallet RPC surface.
//!
//! On first start, the node writes a randomly generated bearer token to
//! `<datadir>/.cookie` (mode `0600`). Clients must send:
//!
//! ```text
//! Authorization: Bearer <token>
//! ```
//!
//! This mirrors Bitcoin Core's cookie-auth model and closes the unauthenticated
//! wallet / control RPC attack surface when the private RPC port is reachable.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

/// Default filename for the RPC auth cookie, relative to the data directory.
pub const DEFAULT_COOKIE_FILENAME: &str = ".cookie";

/// Default cookie path under the node data directory.
pub fn default_cookie_path(datadir: &Path) -> PathBuf {
    datadir.join(DEFAULT_COOKIE_FILENAME)
}

/// Read an existing cookie token from `path`.
pub fn read_cookie(path: &Path) -> anyhow::Result<String> {
    let content = fs::read_to_string(path).map_err(|err| {
        anyhow::anyhow!(
            "failed to read RPC cookie file `{}`: {err}",
            path.display()
        )
    })?;
    let token = content.trim();
    anyhow::ensure!(
        !token.is_empty(),
        "RPC cookie file `{}` is empty",
        path.display()
    );
    // Reject accidental multi-line / corrupted cookies.
    anyhow::ensure!(
        !token.chars().any(|c| c.is_whitespace()),
        "RPC cookie file `{}` contains whitespace; regenerate it",
        path.display()
    );
    Ok(token.to_owned())
}

/// Load the cookie token from `path`, creating a new random cookie if missing.
///
/// The cookie file is created with mode `0600` so only the node user can read
/// the token. Callers should treat the returned token as a secret.
pub fn load_or_create_cookie(path: &Path) -> anyhow::Result<String> {
    if path.exists() {
        return read_cookie(path);
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            anyhow::anyhow!(
                "failed to create RPC cookie directory `{}`: {err}",
                parent.display()
            )
        })?;
    }

    // 32 bytes of entropy encoded as 64 hex characters (two UUID v4s).
    let token = {
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();
        format!("{}{}", a.as_simple(), b.as_simple())
    };

    write_cookie(path, &token)?;
    tracing::info!(
        path = %path.display(),
        "wrote new private-RPC auth cookie (mode 0600)"
    );
    Ok(token)
}

/// Write `token` to `path` with mode `0600`, replacing any existing file.
fn write_cookie(path: &Path, token: &str) -> anyhow::Result<()> {
    // Write via a temp file then rename so a crash cannot leave an empty
    // cookie that subsequent starts would treat as "exists but empty".
    let tmp_path = path.with_extension("cookie.tmp");

    {
        let mut opts = OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp_path).map_err(|err| {
            anyhow::anyhow!(
                "failed to create RPC cookie file `{}`: {err}",
                tmp_path.display()
            )
        })?;
        file.write_all(token.as_bytes()).map_err(|err| {
            anyhow::anyhow!(
                "failed to write RPC cookie file `{}`: {err}",
                tmp_path.display()
            )
        })?;
        file.sync_all().map_err(|err| {
            anyhow::anyhow!(
                "failed to sync RPC cookie file `{}`: {err}",
                tmp_path.display()
            )
        })?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        fs::set_permissions(&tmp_path, perms).map_err(|err| {
            anyhow::anyhow!(
                "failed to set permissions on RPC cookie file `{}`: {err}",
                tmp_path.display()
            )
        })?;
    }

    fs::rename(&tmp_path, path).map_err(|err| {
        anyhow::anyhow!(
            "failed to install RPC cookie file `{}`: {err}",
            path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_cookie_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("thunder_rpc_cookie_{label}_{nanos}"))
    }

    #[test]
    fn load_or_create_writes_and_reloads_same_token() {
        let path = temp_cookie_path("roundtrip");
        drop(fs::remove_file(&path));

        let first = load_or_create_cookie(&path).expect("create cookie");
        assert_eq!(first.len(), 64);
        assert!(path.is_file());

        let second = load_or_create_cookie(&path).expect("reload cookie");
        assert_eq!(first, second);

        drop(fs::remove_file(&path));
    }

    #[test]
    fn read_cookie_rejects_empty_file() {
        let path = temp_cookie_path("empty");
        fs::write(&path, "   ").unwrap();
        let err = read_cookie(&path).unwrap_err().to_string();
        assert!(err.contains("empty"), "err={err}");
        drop(fs::remove_file(&path));
    }

    #[test]
    fn read_cookie_rejects_whitespace_token() {
        let path = temp_cookie_path("ws");
        fs::write(&path, "abc def").unwrap();
        let err = read_cookie(&path).unwrap_err().to_string();
        assert!(err.contains("whitespace"), "err={err}");
        drop(fs::remove_file(&path));
    }
}
