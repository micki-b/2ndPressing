<?php
// 2ndPressing free registration: shared code for index.php and confirm.php.
//
// Secrets and data live outside the website, in ~/secondpressing-private/:
//   signing.key            Ed25519 secret key (base64), signs activation codes
//   postmark-token         Postmark server API token
//   salt                   random; hashes IP addresses for rate limits
//   registrations.sqlite   emails, registration numbers, confirmation tokens
//
// Activation code = "2NDP" + Crockford base32 of
//   payload (7 bytes: version 1, registration number u32 LE, issue day u16 LE)
//   + Ed25519 signature of ("2ndPressing activation v1\0" . payload).
// The app checks it offline (runtime/crates/pressing-license).

declare(strict_types=1);

// MARK: - Settings

const SP_FROM = '2ndPressing <hello@secondpressing.com>';
const SP_SITE = 'https://secondpressing.com';
const SP_TOKEN_HOURS = 48;          // confirmation link: first use
const SP_VIEW_DAYS = 30;            // confirmed link keeps showing the code
const SP_LIMIT_IP_HOUR = 5;         // sign-ups per IP address per hour
const SP_LIMIT_EMAIL_DAY = 3;       // confirmation emails per address per day
const SP_PURGE_DAYS = 7;            // unconfirmed sign-ups are deleted after this

function sp_private(): string {
    // .../public_html/website_x/register -> home directory.
    return dirname(__DIR__, 3) . '/secondpressing-private';
}

// PHP errors go to the private folder, never next to the pages.
ini_set('log_errors', '1');
ini_set('display_errors', '0');
ini_set('error_log', sp_private() . '/php-errors.log');

function sp_secret(string $name): string {
    $v = @file_get_contents(sp_private() . '/' . $name);
    if ($v === false) {
        throw new RuntimeException("missing $name");
    }
    return trim($v);
}

// MARK: - Activation codes

function sp_code(string $secret_key, int $id, int $day): string {
    $payload = chr(1) . pack('V', $id) . pack('v', $day);
    $sig = sodium_crypto_sign_detached("2ndPressing activation v1\0" . $payload, $secret_key);
    return sp_b32($payload . $sig);
}

function sp_b32(string $bytes): string {
    $alphabet = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
    $s = '2NDP';
    $bits = 0;
    $n = 0;
    $k = 0;
    foreach (str_split($bytes) as $ch) {
        $bits = ($bits << 8) | ord($ch);
        $n += 8;
        while ($n >= 5) {
            $n -= 5;
            if ($k % 6 === 0) {
                $s .= '-';
            }
            $s .= $alphabet[($bits >> $n) & 31];
            $k++;
        }
        $bits &= (1 << $n) - 1;
    }
    if ($n > 0) {
        if ($k % 6 === 0) {
            $s .= '-';
        }
        $s .= $alphabet[($bits << (5 - $n)) & 31];
    }
    return $s;
}

function sp_code_for(array $user): string {
    $sk = base64_decode(sp_secret('signing.key'), true);
    return sp_code($sk, (int)$user['id'], (int)$user['issued_day']);
}

// MARK: - Database

function sp_db(): PDO {
    static $db = null;
    if ($db) {
        return $db;
    }
    $db = new PDO('sqlite:' . sp_private() . '/registrations.sqlite', null, null,
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION, PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC]);
    $db->exec('PRAGMA busy_timeout = 3000');
    $db->exec('CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        created INTEGER NOT NULL,
        confirmed INTEGER,
        issued_day INTEGER)');
    $db->exec('CREATE TABLE IF NOT EXISTS tokens (
        hash TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created INTEGER NOT NULL,
        used INTEGER)');
    $db->exec('CREATE TABLE IF NOT EXISTS hits (k TEXT NOT NULL, t INTEGER NOT NULL)');
    // Housekeeping: old rate-limit entries, stale links, unconfirmed sign-ups.
    $now = time();
    $db->prepare('DELETE FROM hits WHERE t < ?')->execute([$now - 86400]);
    $db->prepare('DELETE FROM tokens WHERE created < ?')->execute([$now - SP_VIEW_DAYS * 86400]);
    $db->prepare('DELETE FROM users WHERE confirmed IS NULL AND created < ?')->execute([$now - SP_PURGE_DAYS * 86400]);
    return $db;
}

/// Count this key's recent hits, then record one. True if over the limit.
function sp_limited(string $key, int $window, int $max): bool {
    $db = sp_db();
    $q = $db->prepare('SELECT COUNT(*) FROM hits WHERE k = ? AND t > ?');
    $q->execute([$key, time() - $window]);
    $over = (int)$q->fetchColumn() >= $max;
    $db->prepare('INSERT INTO hits (k, t) VALUES (?, ?)')->execute([$key, time()]);
    return $over;
}

function sp_hash(string $s): string {
    return hash_hmac('sha256', $s, sp_salt());
}

function sp_salt(): string {
    $path = sp_private() . '/salt';
    if (!is_file($path)) {
        $old = umask(077);
        file_put_contents($path, bin2hex(random_bytes(32)));
        umask($old);
    }
    return trim((string)file_get_contents($path));
}

// MARK: - Email

/// Send one email through Postmark. Returns null on success, else a reason.
function sp_send(string $to, string $subject, string $text, string $html): ?string {
    $body = json_encode([
        'From' => SP_FROM,
        'To' => $to,
        'Subject' => $subject,
        'TextBody' => $text,
        'HtmlBody' => $html,
        'MessageStream' => 'outbound',
        'TrackOpens' => false,
        'TrackLinks' => 'None',
    ]);
    $ch = curl_init('https://api.postmarkapp.com/email');
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $body,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 15,
        CURLOPT_HTTPHEADER => [
            'Accept: application/json',
            'Content-Type: application/json',
            'X-Postmark-Server-Token: ' . sp_secret('postmark-token'),
        ],
    ]);
    $res = curl_exec($ch);
    $status = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    $err = curl_error($ch);
    curl_close($ch);
    if ($status === 200) {
        return null;
    }
    $msg = $res ? (json_decode((string)$res, true)['Message'] ?? $res) : $err;
    error_log("postmark $status: $msg");
    return "postmark $status: $msg";
}

// MARK: - Pages

function sp_h(string $s): string {
    return htmlspecialchars($s, ENT_QUOTES | ENT_HTML5, 'UTF-8');
}

function sp_page_start(string $title, bool $index = true): void {
    header('Content-Type: text/html; charset=utf-8');
    header('Cache-Control: private, no-store');
    $robots = $index ? '' : "\n<meta name=\"robots\" content=\"noindex, nofollow\">";
    echo <<<HTML
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{$title}</title>{$robots}
<meta name="description" content="Register your email to get a free activation code for 2ndPressing, the click and pop remover for vinyl.">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/assets/apple-touch-icon.png">
<link rel="stylesheet" href="/assets/site.css?v=6">
<link rel="stylesheet" href="/assets/register.css?v=2">
</head>
<body>

<header class="band top">
  <div class="wrap top-inner">
    <a class="brand" href="/" aria-label="2ndPressing home"><img class="brand-icon" src="/assets/header-icon.svg" alt="" width="36" height="36"><span class="wm">2<span class="nd">nd</span>Pressing</span></a>
    <nav aria-label="Sections"><a href="/">Home</a><a href="/listen/">Listen</a></nav>
  </div>
</header>

<main>
<section class="band get beta-hero">
  <div class="wrap get-inner">
    <div class="get-card beta-card reg-card">
      <div class="get-head">
        <span class="brand compact">2<span class="nd">nd</span>Pressing</span><span class="get-dim">Free activation code</span>
      </div>
      <div class="get-body">

HTML;
}

function sp_page_end(string $after = ''): void {
    echo <<<HTML

      </div>
    </div>
  </div>
</section>
{$after}
</main>

<footer class="band foot">
  <div class="wrap foot-inner">
    <span class="brand compact">2<span class="nd">nd</span>Pressing</span>
    <span>© 2026 Micki Braun · Open source, GPL-3.0</span>
    <span>No cookies, no trackers.</span>
  </div>
</footer>

</body>
</html>

HTML;
}
