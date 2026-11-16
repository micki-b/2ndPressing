<?php
// Free registration, step 1: name and email form. Sends a confirmation
// link; the code is shown when the link is opened (confirm.php), so each
// sign-up costs one email.
declare(strict_types=1);
require __DIR__ . '/_lib.php';

$sent = false;
$error = '';
$email = '';
$first = '';
$last = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = strtolower(trim((string)($_POST['email'] ?? '')));
    [$first, $first_err] = sp_name((string)($_POST['first_name'] ?? ''), 'first name', true);
    [$last, $last_err] = sp_name((string)($_POST['last_name'] ?? ''), 'last name', false);
    $trap = trim((string)($_POST['website'] ?? ''));
    if ($trap !== '') {
        $sent = true; // a bot filled the hidden field: pretend it worked
    } elseif ($first_err || $last_err) {
        $error = (string)($first_err ?? $last_err);
    } elseif (strlen($email) > 254 || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error = 'That email address doesn’t look right. Please check it.';
    } elseif (!isset($_POST['agree'])) {
        $error = 'Please tick the box so we may email you the link.';
    } else {
        try {
            if (sp_limited(sp_hash('ip:' . ($_SERVER['REMOTE_ADDR'] ?? '')), 3600, SP_LIMIT_IP_HOUR)) {
                $error = 'Too many sign-ups from your network. Please try again in an hour.';
            } elseif (sp_limited(sp_hash('em:' . $email), 86400, SP_LIMIT_EMAIL_DAY)) {
                $sent = true; // already sent several today: don't flood the inbox
            } else {
                $db = sp_db();
                $db->prepare('INSERT OR IGNORE INTO users (email, created) VALUES (?, ?)')->execute([$email, time()]);
                $q = $db->prepare('SELECT id FROM users WHERE email = ?');
                $q->execute([$email]);
                $uid = (int)$q->fetchColumn();
                // The latest sign-up's name wins (people fix typos by registering again).
                $db->prepare('UPDATE users SET first_name = ?, last_name = ? WHERE id = ?')
                    ->execute([$first, $last === '' ? null : $last, $uid]);
                $token = rtrim(strtr(base64_encode(random_bytes(24)), '+/', '-_'), '=');
                $db->prepare('INSERT INTO tokens (hash, user_id, created) VALUES (?, ?, ?)')
                    ->execute([hash('sha256', $token), $uid, time()]);
                $link = SP_SITE . '/register/confirm.php?t=' . $token;
                $hours = SP_TOKEN_HOURS;
                // Only the first name goes into the email.
                $text = "Hi $first,\n\n"
                    . "Thanks for signing up for 2ndPressing. Open this link to confirm your email "
                    . "address and get your free activation code:\n\n$link\n\n"
                    . "The link works for $hours hours. If you didn't ask for this, ignore this email; "
                    . "nothing more will be sent.\n\n"
                    . "2ndPressing\nhttps://secondpressing.com\n";
                $l = sp_h($link);
                $f = sp_h($first);
                $html = <<<HTML
<div style="font:15px/1.55 -apple-system,'Segoe UI',sans-serif;color:#2C2A26;max-width:520px">
<p>Hi {$f},</p>
<p>Thanks for signing up for <b>2ndPressing</b>. Confirm your email address to get your free activation code:</p>
<p style="margin:24px 0"><a href="{$l}" style="background:#ECE8DE;border:1px solid #8C8577;border-radius:4px;padding:12px 18px;color:#2C2A26;text-decoration:none;font-weight:600">Confirm and show my code</a></p>
<p style="font-size:13px;color:#6A6455">Or open this link: <a href="{$l}" style="color:#2F3B6B">{$l}</a><br>It works for {$hours} hours. If you didn't ask for this, ignore this email; nothing more will be sent.</p>
<p style="font-size:13px;color:#6A6455">2ndPressing · <a href="https://secondpressing.com" style="color:#2F3B6B">secondpressing.com</a></p>
</div>
HTML;
                $fail = sp_send($email, 'Your 2ndPressing activation code: confirm your email', $text, $html);
                if ($fail) {
                    $error = 'We couldn’t send the email just now. Please try again later.';
                } else {
                    $sent = true;
                }
            }
        } catch (Throwable $e) {
            error_log('register: ' . $e->getMessage());
            $error = 'Something went wrong on our side. Please try again later.';
        }
    }
}

$err_attr = $error ? ' aria-describedby="reg-error"' : '';
sp_page_start('Free activation code · 2ndPressing');
if ($sent) { ?>
        <h1 class="beta-title">Check your inbox</h1>
        <p>Thanks<?= $first !== '' ? ', ' . sp_h($first) : '' ?>. We’ve sent a link to <b><?= sp_h($email) ?></b>. Open it within <?= SP_TOKEN_HOURS ?> hours and your activation code appears right away.</p>
        <p class="small">Nothing after a few minutes? Look in the spam folder, or <a href="/register/">try again</a>.</p>
<?php } else { ?>
        <h1 class="beta-title">Get your free activation code</h1>
        <p>2ndPressing is free. It runs for 14 days without a code; register your email and you get an activation code right away. No payment, no account.</p>
        <form class="reg-form" method="post" action="/register/" novalidate>
          <div class="reg-names">
            <div class="reg-field">
              <label class="reg-label" for="first_name">First name</label>
              <input class="reg-input" id="first_name" name="first_name" type="text" autocomplete="given-name" maxlength="<?= SP_NAME_MAX ?>" required value="<?= sp_h($first) ?>"<?= $err_attr ?>>
            </div>
            <div class="reg-field">
              <label class="reg-label" for="last_name">Last name <span class="reg-opt">optional</span></label>
              <input class="reg-input" id="last_name" name="last_name" type="text" autocomplete="family-name" maxlength="<?= SP_NAME_MAX ?>" value="<?= sp_h($last) ?>">
            </div>
          </div>
          <div class="reg-field">
            <label class="reg-label" for="email">Email address</label>
            <input class="reg-input" id="email" name="email" type="email" autocomplete="email" maxlength="254" required value="<?= sp_h($email) ?>"<?= $err_attr ?>>
          </div>
          <div class="reg-hp" aria-hidden="true"><label>Leave this empty <input name="website" type="text" tabindex="-1" autocomplete="off"></label></div>
          <label class="reg-agree"><input type="checkbox" name="agree" value="1"<?= isset($_POST['agree']) ? ' checked' : '' ?>> <span>Email me the confirmation link and, rarely, news about new 2ndPressing versions. I can ask to be deleted any time.</span></label>
<?php if ($error) { ?>
          <p class="reg-error" id="reg-error" role="alert"><?= sp_h($error) ?></p>
<?php } ?>
          <button class="key primary" type="submit"><span class="lamp" aria-hidden="true"></span>Send me the link</button>
        </form>
        <p class="small reg-privacy">What’s kept: your name, your email address and the date you registered, used only for 2ndPressing emails, never shared or sold. To be deleted, write to <a href="mailto:hello@secondpressing.com">hello@secondpressing.com</a>.</p>
<?php }
sp_page_end();
