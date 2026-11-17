//! Check an activation code: `cargo run -p pressing-license --example check -- "2NDP-…"`
//! (or pipe text containing one on stdin). Prints the registration number
//! and issue date, or why the code isn't valid.
use std::io::Read;

fn main() {
    let mut text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if text.trim().is_empty() {
        std::io::stdin().read_to_string(&mut text).ok();
    }
    match pressing_license::parse(&text) {
        Ok(a) => println!("valid: registration #{} issued {}", a.id, a.issued()),
        Err(e) => {
            eprintln!("not valid: {}", e.message());
            std::process::exit(1);
        }
    }
}
