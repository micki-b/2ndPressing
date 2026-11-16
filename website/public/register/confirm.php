<?php
// Free registration, step 2: the emailed link. Confirms the address, issues
// the activation code (once per address; the same code every time) and
// shows it. The link keeps showing the code for SP_VIEW_DAYS days.
declare(strict_types=1);
require __DIR__ . '/_lib.php';

$token = (string)($_GET['t'] ?? '');
$user = null;
try {
    if (preg_match('/^[A-Za-z0-9_-]{20,64}$/', $token)) {
        $db = sp_db();
        $q = $db->prepare('SELECT t.created, t.used, u.* FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.hash = ?');
        $q->execute([hash('sha256', $token)]);
        $row = $q->fetch();
        $now = time();
        if ($row && ($row['used'] !== null || $row['created'] > $now - SP_TOKEN_HOURS * 3600)) {
            if ($row['used'] === null) {
                $db->prepare('UPDATE tokens SET used = ? WHERE hash = ?')->execute([$now, hash('sha256', $token)]);
            }
            if ($row['confirmed'] === null) {
                $db->prepare('UPDATE users SET confirmed = ?, issued_day = ? WHERE id = ?')
                    ->execute([$now, intdiv($now, 86400), $row['id']]);
                $row['issued_day'] = intdiv($now, 86400);
            }
            $user = $row;
        }
    }
    $code = $user ? sp_code_for($user) : '';
} catch (Throwable $e) {
    error_log('confirm: ' . $e->getMessage());
    $user = null;
    $code = '';
}

if ($user && isset($_GET['download'])) {
    header('Content-Type: text/plain; charset=utf-8');
    header('Content-Disposition: attachment; filename="2ndPressing-activation-code.txt"');
    header('Cache-Control: private, no-store');
    echo "2ndPressing activation code\r\n\r\n$code\r\n\r\n"
        . "To activate: open 2ndPressing (the app or a plugin), click the FREE TRIAL tag at the bottom right\r\n"
        . "of the faceplate, copy the code above and press PASTE CODE.\r\n\r\nhttps://secondpressing.com\r\n";
    exit;
}

if (!$user) {
    http_response_code(404);
}
sp_page_start('Your activation code · 2ndPressing', false);
if (!$user) { ?>
        <h1 class="beta-title">This link has expired</h1>
        <p>Confirmation links work for <?= SP_TOKEN_HOURS ?> hours. <a href="/register/">Register again</a> with the same email address and you get a new link, with the same code if you already had one.</p>
<?php } else { ?>
        <h1 class="beta-title">Your activation code</h1>
        <p>Thank you for registering. This code activates 2ndPressing on your computers, the app and every plugin format.</p>
        <p class="reg-code" id="code"><?= sp_h($code) ?></p>
        <div class="reg-actions">
          <button class="key primary" type="button" id="copy" hidden><span class="lamp" aria-hidden="true"></span><span class="lbl">Copy code</span></button>
          <a class="key" href="?t=<?= sp_h($token) ?>&amp;download=1"><span class="lamp" aria-hidden="true"></span>Save as text file</a>
        </div>
        <h2 class="reg-h2">To activate</h2>
        <ol class="beta-list">
          <li><a href="/beta/">Download and install 2ndPressing</a> for Mac or Windows, if you haven’t yet.</li>
          <li>Open the app, or the plugin in your DAW. At the bottom right of the faceplate, click <b>FREE TRIAL · … DAYS LEFT</b>.</li>
          <li>Copy the code above, then press <b>PASTE CODE</b>. Done: the check is offline, nothing is sent anywhere.</li>
        </ol>
        <p class="small">Keep the code: you need it again on a new computer. This page keeps working for <?= SP_VIEW_DAYS ?> days, and registering the same email again always gives the same code.</p>
<?php }
sp_page_end($user ? '<script src="/register/copy.js"></script>' : '');
