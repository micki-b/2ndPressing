//! Free activation codes and the 14-day trial.
//!
//! 2ndPressing is free. People register an email address on
//! secondpressing.com and get an activation code: a 7-byte payload (format
//! version, registration number, issue day) signed with the website's
//! Ed25519 key. The check is offline: the app and plugins never contact the
//! server. Without a code, everything works for [`TRIAL_DAYS`] days from the
//! first run, then audio passes through unprocessed until a code is entered.
//!
//! State lives next to the app's settings, in
//! `<config dir>/2ndPressing/license.json`, shared by the app and every
//! plugin format. When that file can't be read or written (a sandboxed
//! host, for example) the trial is treated as running: the check fails open.
//!
//! The source is GPL: this is a registration step, not copy protection.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

// MARK: - Constants

/// Days the software works without an activation code.
pub const TRIAL_DAYS: u32 = 14;

/// Where people get a code.
pub const REGISTER_URL: &str = "https://secondpressing.com/register/";

/// The website's public key (its private half signs codes on the server).
const PUBLIC_KEY: [u8; 32] = [
    0x59, 0x84, 0xf2, 0x5f, 0x14, 0x82, 0x92, 0x1b, 0x5b, 0xe7, 0x6f, 0x48, 0xdd, 0xb2, 0x11, 0xab,
    0x78, 0x60, 0x3b, 0x28, 0xa1, 0x0c, 0x46, 0x61, 0xea, 0x15, 0x38, 0x73, 0xb8, 0x28, 0xea, 0xa2,
];

/// Signed message = this prefix + payload (keeps the signature specific to
/// 2ndPressing codes). The server uses the same prefix.
const DOMAIN: &[u8] = b"2ndPressing activation v1\0";
/// Printed before the code; also how a code is found inside pasted text.
pub const PREFIX: &str = "2NDP";
const VERSION: u8 = 1;
const PAYLOAD: usize = 7;
const SIG: usize = 64;
const CODE_BYTES: usize = PAYLOAD + SIG;
/// Base32 characters in a code (71 bytes, 5 bits per character).
const CODE_CHARS: usize = (CODE_BYTES * 8).div_ceil(5);
/// Crockford base32: no I, L, O or U, so codes survive retyping.
const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

// MARK: - Codes

/// A verified activation code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Activation {
    /// Registration number on the website.
    pub id: u32,
    /// Day the code was issued (days since 1970-01-01, UTC).
    pub issued_day: u16,
}

impl Activation {
    /// Issue date as `YYYY-MM-DD`.
    pub fn issued(&self) -> String {
        let (y, m, d) = civil_from_days(i64::from(self.issued_day));
        format!("{y:04}-{m:02}-{d:02}")
    }
}

/// Why a code was not accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeError {
    /// Nothing that looks like a code.
    NotFound,
    /// Too short, or not base32.
    Malformed,
    /// Made for a newer version of 2ndPressing.
    UnknownVersion,
    /// The signature doesn't match: mistyped or not a real code.
    Invalid,
    /// The code is valid but couldn't be saved.
    NotSaved,
}

impl CodeError {
    pub fn message(self) -> &'static str {
        match self {
            CodeError::NotFound => "No activation code found on the clipboard. Copy the whole code first.",
            CodeError::Malformed => "That code is incomplete. Copy the whole code, from 2NDP to the end.",
            CodeError::UnknownVersion => "That code is for a newer 2ndPressing. Please update.",
            CodeError::Invalid => "That code isn't valid. Copy it again from the email or the website.",
            CodeError::NotSaved => "The code is valid but couldn't be saved on this computer.",
        }
    }
}

/// Find, decode and verify a code anywhere in `text` (a pasted email is
/// fine: the code is found by its `2NDP` prefix).
pub fn parse(text: &str) -> Result<Activation, CodeError> {
    parse_with(text, &PUBLIC_KEY)
}

fn parse_with(text: &str, key: &[u8; 32]) -> Result<Activation, CodeError> {
    let bytes = decode(text)?;
    let (payload, sig) = bytes.split_at(PAYLOAD);
    if payload[0] != VERSION {
        return Err(CodeError::UnknownVersion);
    }
    let vk = VerifyingKey::from_bytes(key).map_err(|_| CodeError::Invalid)?;
    let sig = Signature::from_bytes(sig.try_into().map_err(|_| CodeError::Malformed)?);
    let mut msg = DOMAIN.to_vec();
    msg.extend_from_slice(payload);
    vk.verify_strict(&msg, &sig).map_err(|_| CodeError::Invalid)?;
    Ok(Activation {
        id: u32::from_le_bytes(payload[1..5].try_into().unwrap()),
        issued_day: u16::from_le_bytes(payload[5..7].try_into().unwrap()),
    })
}

/// The code's bytes, from text that contains `2NDP` followed by the code
/// (dashes, spaces and line breaks inside it are ignored), or from text that
/// is just the code.
fn decode(text: &str) -> Result<[u8; CODE_BYTES], CodeError> {
    let upper = text.to_ascii_uppercase();
    let start = match upper.find(PREFIX) {
        Some(i) => i + PREFIX.len(),
        None if upper.chars().any(|c| c.is_ascii_alphanumeric()) => 0,
        None => return Err(CodeError::NotFound),
    };
    let mut bits: u64 = 0;
    let mut nbits = 0;
    let mut out = Vec::with_capacity(CODE_BYTES);
    let mut chars = 0;
    for c in upper[start..].chars() {
        if chars == CODE_CHARS {
            break;
        }
        let v = match c {
            '-' | ' ' | '\t' | '\r' | '\n' => continue,
            'O' => 0,
            'I' | 'L' => 1,
            _ => match ALPHABET.iter().position(|&a| a as char == c) {
                Some(v) => v as u64,
                None => break,
            },
        };
        chars += 1;
        bits = (bits << 5) | v;
        nbits += 5;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
            bits &= (1 << nbits) - 1;
        }
    }
    if chars < CODE_CHARS || out.len() < CODE_BYTES {
        return Err(CodeError::Malformed);
    }
    out.truncate(CODE_BYTES);
    Ok(out.try_into().unwrap())
}

/// Text form of a code: `2NDP-` then base32 in groups of six.
pub fn encode(bytes: &[u8]) -> String {
    let mut s = String::from(PREFIX);
    let (mut bits, mut nbits, mut n) = (0u64, 0, 0);
    let mut push = |v: u64, s: &mut String| {
        if n % 6 == 0 {
            s.push('-');
        }
        s.push(ALPHABET[v as usize] as char);
        n += 1;
    };
    for &b in bytes {
        bits = (bits << 8) | u64::from(b);
        nbits += 8;
        while nbits >= 5 {
            nbits -= 5;
            push((bits >> nbits) & 31, &mut s);
        }
        bits &= (1 << nbits) - 1;
    }
    if nbits > 0 {
        push((bits << (5 - nbits)) & 31, &mut s);
    }
    s
}

// MARK: - Status

/// What the software may do right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Registered(Activation),
    Trial { days_left: u32 },
    /// Trial over, no code: audio passes through unprocessed.
    Expired,
}

impl Status {
    pub fn may_process(self) -> bool {
        !matches!(self, Status::Expired)
    }
}

/// `license.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LicenseFile {
    /// The activation code as entered (verified again on every check).
    pub code: Option<String>,
    /// First run, days since 1970-01-01.
    pub trial_start: Option<u32>,
}

/// Status on `today`; starts the trial clock if it hasn't started (the
/// caller saves `file` when this returns `true` as the second value).
pub fn status_at(file: &mut LicenseFile, today: u32) -> (Status, bool) {
    if let Some(a) = file.code.as_deref().and_then(|c| parse(c).ok()) {
        return (Status::Registered(a), false);
    }
    let mut changed = false;
    let start = *file.trial_start.get_or_insert_with(|| {
        changed = true;
        today
    });
    let used = today.saturating_sub(start);
    let status = if used < TRIAL_DAYS {
        Status::Trial { days_left: TRIAL_DAYS - used }
    } else {
        Status::Expired
    };
    (status, changed)
}

/// Read the license file, start the trial if needed, and return the status.
/// Fails open: if the file can't be read or written, the trial is running.
pub fn check() -> Status {
    let Some(path) = file_path() else {
        return Status::Trial { days_left: TRIAL_DAYS };
    };
    let mut file = match std::fs::read(&path) {
        Ok(b) => serde_json::from_slice(&b).unwrap_or_default(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => LicenseFile::default(),
        Err(_) => return Status::Trial { days_left: TRIAL_DAYS },
    };
    let (status, changed) = status_at(&mut file, today());
    if changed && save(&path, &file).is_err() {
        return Status::Trial { days_left: TRIAL_DAYS };
    }
    status
}

/// Verify a code from pasted text and save it.
pub fn activate(text: &str) -> Result<Activation, CodeError> {
    let a = parse(text)?;
    let path = file_path().ok_or(CodeError::NotSaved)?;
    let mut file: LicenseFile = std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    file.code = Some(canonical(text).unwrap_or_else(|| text.trim().to_string()));
    save(&path, &file).map_err(|_| CodeError::NotSaved)?;
    Ok(a)
}

/// The code alone, in its printed form.
fn canonical(text: &str) -> Option<String> {
    decode(text).ok().map(|b| encode(&b))
}

fn save(path: &std::path::Path, file: &LicenseFile) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let json = serde_json::to_vec_pretty(file).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Days since 1970-01-01 (UTC).
pub fn today() -> u32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (secs / 86_400) as u32
}

// MARK: - Paths

/// `<config dir>/2ndPressing`: Application Support on macOS, %APPDATA% on
/// Windows, `$XDG_CONFIG_HOME` (or ~/.config) elsewhere.
pub fn config_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let base = if cfg!(target_os = "macos") {
        home.map(|h| h.join("Library/Application Support"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home.map(|h| h.join(".config")))
    }?;
    Some(base.join("2ndPressing"))
}

fn file_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("license.json"))
}

/// Gregorian date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

// MARK: - Tests

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    /// Test key (seed 1..=32), shared with the PHP test vector below.
    fn key() -> SigningKey {
        let seed: [u8; 32] = core::array::from_fn(|i| i as u8 + 1);
        SigningKey::from_bytes(&seed)
    }

    fn make(id: u32, day: u16) -> String {
        let mut payload = vec![VERSION];
        payload.extend_from_slice(&id.to_le_bytes());
        payload.extend_from_slice(&day.to_le_bytes());
        let mut msg = DOMAIN.to_vec();
        msg.extend_from_slice(&payload);
        payload.extend_from_slice(&key().sign(&msg).to_bytes());
        encode(&payload)
    }

    fn pk() -> [u8; 32] {
        key().verifying_key().to_bytes()
    }

    #[test]
    fn round_trip() {
        let code = make(1234, 20_408);
        assert!(code.starts_with("2NDP-"));
        assert_eq!(code.chars().filter(|c| c.is_ascii_alphanumeric()).count(), 4 + CODE_CHARS);
        let a = parse_with(&code, &pk()).unwrap();
        assert_eq!(a, Activation { id: 1234, issued_day: 20_408 });
        assert_eq!(a.issued(), "2025-11-16");
    }

    #[test]
    fn found_inside_pasted_email_with_line_breaks_and_lowercase() {
        let code = make(7, 20_773);
        let (a, b) = code.split_at(40);
        let text = format!("Hi,\n\nYour activation code:\n\n  {}\n{}\n\nEnjoy!", a.to_lowercase(), b);
        assert_eq!(parse_with(&text, &pk()).unwrap().id, 7);
    }

    #[test]
    fn rejects_tampering_truncation_and_other_keys() {
        let code = make(42, 20_773);
        // Change one character of the payload (the id).
        let mut chars: Vec<char> = code.chars().collect();
        let i = 9; // inside the registration number
        chars[i] = if chars[i] == 'A' { 'B' } else { 'A' };
        let bad: String = chars.into_iter().collect();
        assert_eq!(parse_with(&bad, &pk()), Err(CodeError::Invalid));
        assert_eq!(parse_with(&code[..60], &pk()), Err(CodeError::Malformed));
        assert_eq!(parse_with("", &pk()), Err(CodeError::NotFound));
        // The real key doesn't accept test-key codes.
        assert_eq!(parse(&code), Err(CodeError::Invalid));
    }

    /// Made by the website's PHP (`sodium_crypto_sign_detached`) with the
    /// test key, so the two implementations agree byte for byte.
    #[test]
    fn php_test_vector() {
        const PHP: &str = include_str!("../tests/php-vector.txt");
        let a = parse_with(PHP, &pk()).unwrap();
        assert_eq!(a, Activation { id: 31_337, issued_day: 20_773 });
        assert_eq!(PHP.trim(), make(31_337, 20_773));
    }

    #[test]
    fn trial_counts_days_then_expires() {
        let mut f = LicenseFile::default();
        assert_eq!(status_at(&mut f, 100), (Status::Trial { days_left: 14 }, true));
        assert_eq!(status_at(&mut f, 100), (Status::Trial { days_left: 14 }, false));
        assert_eq!(status_at(&mut f, 113).0, Status::Trial { days_left: 1 });
        assert_eq!(status_at(&mut f, 114).0, Status::Expired);
        assert!(!Status::Expired.may_process());
        // Clock set back: still in the trial, not negative.
        assert_eq!(status_at(&mut f, 90).0, Status::Trial { days_left: 14 });
        // A code that doesn't verify doesn't unlock.
        f.code = Some(make(1, 1));
        assert_eq!(status_at(&mut f, 200).0, Status::Expired);
    }

    #[test]
    fn dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_408), (2025, 11, 16));
        assert_eq!(civil_from_days(20_773), (2026, 11, 16));
    }
}
