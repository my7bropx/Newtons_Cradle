//! Newton's Cradle — a physics toy in Rust with dynamically synthesized sound.
//!
//! The cradle is modeled as a row of equal-mass pendulums whose pivots are
//! spaced exactly one ball-diameter apart, so the balls just touch at rest.
//! When two adjacent balls make contact while approaching, we exchange their
//! angular velocities (the textbook result for a head-on elastic collision of
//! equal masses). Iterating that exchange across the row reproduces the iconic
//! "one in, one out" behavior.
//!
//! Everything you hear is *synthesized*, not loaded from disk:
//!   * a 2-D bank of metallic "clack" waveforms (pitch x stereo pan), each run
//!     through a small synthesized room reverb (early reflections + a Schroeder
//!     diffuse tail) so it sounds like it's in an empty room;
//!   * a bank of airy "whoosh" sweeps played as a ball rushes through the
//!     bottom of its arc (Doppler-style rise/fall + stereo pan).
//! Impacts kick up a dust/spark puff and shake the screen on big hits. Sliders
//! tweak air damping, room size, and time scale; volume/mute/pause are on keys.
//! Settings persist in ~/.newtons_cradle.cfg between runs.

use macroquad::audio::{load_sound_from_bytes, play_sound, PlaySoundParams, Sound};
use macroquad::prelude::*;
use macroquad::rand::gen_range;
use std::f32::consts::{FRAC_PI_2, TAU};

// ----- simulation tuning -------------------------------------------------
const BALL_R: f32 = 28.0; // ball radius (px)
const STRING_L: f32 = 250.0; // pendulum length, pivot -> ball center (px)
const GRAVITY: f32 = 2600.0; // gravity in screen units (px/s^2)
const SUBSTEPS: u32 = 16; // physics substeps per frame (stability)
const PIVOT_Y: f32 = 130.0; // y of the support bar
const TRAIL_LEN: usize = 22; // history length of the motion trail

// ----- screen shake ------------------------------------------------------
const SHAKE_MIN: f32 = 600.0; // impact speed (px/s) before the screen shakes
const SHAKE_MAX_PX: f32 = 16.0; // max shake amplitude (px)
const SHAKE_DECAY: f32 = 70.0; // shake amplitude lost per second

// ----- audio tuning ------------------------------------------------------
const SAMPLE_RATE: u32 = 44_100;
const PITCH_STEPS: usize = 12; // clack pitch variations
const PAN_STEPS: usize = 5; // pre-rendered pan positions (L..R), shared
const FREQ_LO: f32 = 720.0; // base pitch of the softest clack
const FREQ_HI: f32 = 1500.0; // base pitch of the hardest clack
const SPEED_FULL: f32 = 1300.0; // impact speed (px/s) mapped to max pitch/vol
const MIN_AUDIBLE: f32 = 40.0; // ignore impacts gentler than this (px/s)
const REVERB_TAIL: f32 = 0.55; // base room tail (scaled by room size) (s)

// whoosh (air-sweep) bank
const WHOOSH_STEPS: usize = 5;
const WHOOSH_FC_LO: f32 = 380.0;
const WHOOSH_FC_HI: f32 = 1100.0;
const WHOOSH_MIN: f32 = 260.0;
const WHOOSH_FULL: f32 = 1500.0;

// Schroeder reverb base tunings (delay samples @ 44.1 kHz, feedback gain).
const COMBS: [(usize, f32); 4] = [(1116, 0.80), (1188, 0.79), (1277, 0.81), (1356, 0.78)];
const ALLPASS: [(usize, f32); 2] = [(556, 0.5), (441, 0.5)];
const EARLY: [(f32, f32); 3] = [(0.071, 0.45), (0.113, 0.30), (0.161, 0.20)];

struct Ball {
    theta: f32,
    omega: f32,
    pivot_x: f32,
}
impl Ball {
    fn pos(&self) -> Vec2 {
        vec2(
            self.pivot_x + STRING_L * self.theta.sin(),
            PIVOT_Y + STRING_L * self.theta.cos(),
        )
    }
}

struct Particle {
    pos: Vec2,
    vel: Vec2,
    life: f32,
    life0: f32,
    size: f32,
    spark: bool,
}

/// Persisted, user-tweakable settings. Stored as plain `key=value` text.
#[derive(Clone, PartialEq)]
struct Config {
    damping: f32,
    room: f32,
    time_scale: f32,
    volume: f32,
    mute: bool,
    n_balls: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self { damping: 0.04, room: 1.0, time_scale: 1.0, volume: 0.8, mute: false, n_balls: 5 }
    }
}
impl Config {
    fn clamped(mut self) -> Self {
        self.damping = self.damping.clamp(0.0, 0.25);
        self.room = self.room.clamp(0.4, 1.8);
        self.time_scale = self.time_scale.clamp(0.05, 2.0);
        self.volume = self.volume.clamp(0.0, 1.0);
        self.n_balls = self.n_balls.clamp(3, 7);
        self
    }
}

fn config_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home).join(".newtons_cradle.cfg")
}
fn load_config() -> Config {
    let mut c = Config::default();
    if let Ok(text) = std::fs::read_to_string(config_path()) {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                let (k, v) = (k.trim(), v.trim());
                match k {
                    "damping" => c.damping = v.parse().unwrap_or(c.damping),
                    "room" => c.room = v.parse().unwrap_or(c.room),
                    "time" => c.time_scale = v.parse().unwrap_or(c.time_scale),
                    "volume" => c.volume = v.parse().unwrap_or(c.volume),
                    "mute" => c.mute = v == "1" || v == "true",
                    "balls" => c.n_balls = v.parse().unwrap_or(c.n_balls),
                    _ => {}
                }
            }
        }
    }
    c.clamped()
}
fn save_config(c: &Config) {
    let s = format!(
        "damping={}\nroom={}\ntime={}\nvolume={}\nmute={}\nballs={}\n",
        c.damping, c.room, c.time_scale, c.volume, c.mute as u8, c.n_balls
    );
    let _ = std::fs::write(config_path(), s);
}

// ===== audio synthesis ===================================================

fn encode_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bytes_per_sample = 2u32;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let block_align = channels as u32 * bytes_per_sample;
    let byte_rate = sample_rate * block_align;
    let mut v = Vec::with_capacity(44 + data_len as usize);

    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVE");
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&byte_rate.to_le_bytes());
    v.extend_from_slice(&(block_align as u16).to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let i = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        v.extend_from_slice(&i.to_le_bytes());
    }
    v
}

fn pan_to_stereo(mono: &[f32], pan: f32) -> Vec<f32> {
    let angle = (pan.clamp(-1.0, 1.0) * 0.5 + 0.5) * FRAC_PI_2;
    let (lg, rg) = (angle.cos(), angle.sin());
    let mut out = Vec::with_capacity(mono.len() * 2);
    for &s in mono {
        out.push(s * lg);
        out.push(s * rg);
    }
    out
}

fn normalize(buf: &mut [f32], target: f32) {
    let peak = buf.iter().fold(1e-6_f32, |m, &s| m.max(s.abs()));
    let g = target / peak;
    for s in buf {
        *s *= g;
    }
}

fn noise_gen(mut seed: u32) -> impl FnMut() -> f32 {
    move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn render_dry_click(base_freq: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * 0.13) as usize;
    let partials = [
        (1.00_f32, 1.00_f32, 0.050_f32),
        (2.76, 0.60, 0.038),
        (5.40, 0.32, 0.026),
        (8.93, 0.16, 0.018),
    ];
    let mut noise = noise_gen(0x9E37_79B9 ^ (base_freq as u32).wrapping_mul(2_654_435_761));
    let mut out = vec![0.0_f32; n];
    for (i, sample) in out.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        let mut v = 0.0;
        for &(ratio, amp, tau) in &partials {
            v += amp * (TAU * base_freq * ratio * t).sin() * (-t / tau).exp();
        }
        v += noise() * (-t / 0.0018).exp() * 0.7;
        *sample = v;
    }
    out
}

fn render_room_click(base_freq: f32, rs: f32) -> Vec<f32> {
    let dry = render_dry_click(base_freq);
    let tail = REVERB_TAIL * (0.5 + 0.85 * rs);
    let total = dry.len() + (SAMPLE_RATE as f32 * tail) as usize;

    let mut x = vec![0.0_f32; total];
    x[..dry.len()].copy_from_slice(&dry);

    let mut wet = vec![0.0_f32; total];
    for &(d, g) in &COMBS {
        let d = ((d as f32 * (0.7 + 0.5 * rs)) as usize).max(1);
        let g = (g * (0.90 + 0.07 * rs)).min(0.93);
        let mut buf = vec![0.0_f32; total];
        for n in 0..total {
            let fb = if n >= d { buf[n - d] } else { 0.0 };
            buf[n] = x[n] + g * fb;
        }
        for n in 0..total {
            wet[n] += buf[n] * 0.25;
        }
    }
    for &(d, g) in &ALLPASS {
        let d = ((d as f32 * (0.8 + 0.3 * rs)) as usize).max(1);
        let mut v = vec![0.0_f32; total];
        let mut out = vec![0.0_f32; total];
        for n in 0..total {
            let vd = if n >= d { v[n - d] } else { 0.0 };
            v[n] = wet[n] + g * vd;
            out[n] = -g * v[n] + vd;
        }
        wet = out;
    }

    let mut mix = vec![0.0_f32; total];
    for n in 0..total {
        mix[n] = x[n] + 0.9 * wet[n];
    }
    for &(dt, g) in &EARLY {
        let d = ((dt * (0.6 + 0.8 * rs)) * SAMPLE_RATE as f32) as usize;
        for n in d..total {
            mix[n] += g * x[n - d];
        }
    }
    normalize(&mut mix, 0.9);
    mix
}

fn render_whoosh(center: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * 0.22) as usize;
    let mut noise = noise_gen(0x1234_5678 ^ (center as u32).wrapping_mul(40_503));
    let (mut x1, mut x2, mut y1, mut y2) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let q = 1.4_f32;
    let mut out = vec![0.0_f32; n];
    for (i, sample) in out.iter_mut().enumerate() {
        let u = i as f32 / n as f32;
        let sweep = 1.0 + 0.30 * (TAU * 0.5 * u).sin() - 0.15 * u;
        let fc = (center * sweep).clamp(120.0, 6000.0);
        let w0 = TAU * fc / SAMPLE_RATE as f32;
        let (sw, cw) = (w0.sin(), w0.cos());
        let alpha = sw / (2.0 * q);
        let (b0, b2, a0, a1, a2) = (alpha, -alpha, 1.0 + alpha, -2.0 * cw, 1.0 - alpha);
        let xn = noise();
        let yn = (b0 * xn + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        x2 = x1;
        x1 = xn;
        y2 = y1;
        y1 = yn;
        let env = (std::f32::consts::PI * u).sin().powf(1.4) * (1.0 - 0.25 * u);
        *sample = yn * env;
    }
    normalize(&mut out, 0.85);
    out
}

async fn build_clack_bank(rs: f32) -> Vec<Sound> {
    let mut bank = Vec::with_capacity(PITCH_STEPS * PAN_STEPS);
    for pi in 0..PITCH_STEPS {
        let f = FREQ_LO + (FREQ_HI - FREQ_LO) * pi as f32 / (PITCH_STEPS as f32 - 1.0);
        let room = render_room_click(f, rs);
        for pj in 0..PAN_STEPS {
            let pan = (pj as f32 / (PAN_STEPS as f32 - 1.0)) * 2.0 - 1.0;
            let wav = encode_wav(&pan_to_stereo(&room, pan), SAMPLE_RATE, 2);
            bank.push(load_sound_from_bytes(&wav).await.unwrap());
        }
    }
    bank
}

async fn build_whoosh_bank() -> Vec<Sound> {
    let mut bank = Vec::with_capacity(WHOOSH_STEPS * PAN_STEPS);
    for pi in 0..WHOOSH_STEPS {
        let fc = WHOOSH_FC_LO + (WHOOSH_FC_HI - WHOOSH_FC_LO) * pi as f32 / (WHOOSH_STEPS as f32 - 1.0);
        let mono = render_whoosh(fc);
        for pj in 0..PAN_STEPS {
            let pan = (pj as f32 / (PAN_STEPS as f32 - 1.0)) * 2.0 - 1.0;
            let wav = encode_wav(&pan_to_stereo(&mono, pan), SAMPLE_RATE, 2);
            bank.push(load_sound_from_bytes(&wav).await.unwrap());
        }
    }
    bank
}

// ===== helpers ===========================================================

fn step_index(t: f32, steps: usize) -> usize {
    ((t.clamp(0.0, 1.0) * (steps as f32 - 1.0)).round() as usize).min(steps - 1)
}
fn pan_index(x: f32, sw: f32, n_balls: usize) -> usize {
    let span = (n_balls as f32) * BALL_R;
    let pan = ((x - sw * 0.5) / span).clamp(-1.0, 1.0);
    step_index(pan * 0.5 + 0.5, PAN_STEPS)
}
/// Master gain applied to every voice (volume + mute).
fn gain(base: f32, volume: f32, mute: bool) -> f32 {
    if mute {
        0.0
    } else {
        base * volume
    }
}

fn make_balls(n: usize, screen_w: f32) -> Vec<Ball> {
    let spacing = 2.0 * BALL_R;
    let center_x = screen_w * 0.5;
    (0..n)
        .map(|i| Ball {
            theta: 0.0,
            omega: 0.0,
            pivot_x: center_x + (i as f32 - (n as f32 - 1.0) / 2.0) * spacing,
        })
        .collect()
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Newton's Cradle".to_owned(),
        window_width: 1000,
        window_height: 640,
        high_dpi: true,
        ..Default::default()
    }
}

// ----- on-screen sliders (top-left) --------------------------------------
const SLIDER_X: f32 = 18.0;
const SLIDER_W: f32 = 200.0;
struct SliderDef {
    y: f32,
    lo: f32,
    hi: f32,
    label: &'static str,
}
const SLIDERS: [SliderDef; 3] = [
    SliderDef { y: 92.0, lo: 0.0, hi: 0.25, label: "air damping" },
    SliderDef { y: 124.0, lo: 0.4, hi: 1.8, label: "room size" },
    SliderDef { y: 156.0, lo: 0.05, hi: 2.0, label: "time scale" },
];

fn slider_hit(i: usize, m: Vec2) -> bool {
    let s = &SLIDERS[i];
    m.x >= SLIDER_X - 10.0 && m.x <= SLIDER_X + SLIDER_W + 10.0 && (m.y - s.y).abs() <= 12.0
}
fn slider_value_from_mouse(i: usize, mx: f32) -> f32 {
    let s = &SLIDERS[i];
    let frac = ((mx - SLIDER_X) / SLIDER_W).clamp(0.0, 1.0);
    s.lo + frac * (s.hi - s.lo)
}

/// View-model passed to the renderer for the HUD.
struct Ui {
    damping: f32,
    room: f32,
    time_scale: f32,
    volume: f32,
    mute: bool,
    paused: bool,
}

// ===== main ==============================================================

#[macroquad::main(window_conf)]
async fn main() {
    let cfg = load_config();
    let mut damping = cfg.damping;
    let mut room_size = cfg.room;
    let mut time_scale = cfg.time_scale;
    let mut volume = cfg.volume;
    let mut mute = cfg.mute;
    let mut n_balls = cfg.n_balls;
    let mut paused = false;
    let mut last_saved = cfg;

    let mut clack_bank = build_clack_bank(room_size).await;
    let whoosh_bank = build_whoosh_bank().await;

    let mut balls = make_balls(n_balls, screen_width());
    let mut trails: Vec<Vec<Vec2>> = vec![Vec::new(); n_balls];
    let mut prev_theta: Vec<f32> = balls.iter().map(|b| b.theta).collect();
    let mut particles: Vec<Particle> = Vec::new();
    if let Some(b) = balls.first_mut() {
        b.theta = -0.9;
    }

    let mut dragging: Option<usize> = None;
    let mut active_slider: Option<usize> = None;
    let mut need_rebuild = false;
    let mut shake = 0.0_f32;

    loop {
        let sw = screen_width();
        let sh = screen_height();

        if need_rebuild {
            clear_background(Color::new(0.08, 0.09, 0.12, 1.0));
            draw_text("updating room acoustics...", sw * 0.5 - 150.0, sh * 0.5, 28.0, Color::new(0.85, 0.87, 0.95, 1.0));
            next_frame().await;
            clack_bank = build_clack_bank(room_size).await;
            need_rebuild = false;
            continue;
        }

        // ----- input ------------------------------------------------------
        if is_key_pressed(KeyCode::Escape) {
            save_config(&Config { damping, room: room_size, time_scale, volume, mute, n_balls });
            break;
        }
        if is_key_pressed(KeyCode::R) {
            balls = make_balls(n_balls, sw);
            balls[0].theta = -0.9;
            dragging = None;
        }
        if is_key_pressed(KeyCode::Up) && n_balls < 7 {
            n_balls += 1;
            balls = make_balls(n_balls, sw);
            dragging = None;
        }
        if is_key_pressed(KeyCode::Down) && n_balls > 3 {
            n_balls -= 1;
            balls = make_balls(n_balls, sw);
            dragging = None;
        }
        for (key, k) in [(KeyCode::Key1, 1usize), (KeyCode::Key2, 2), (KeyCode::Key3, 3), (KeyCode::Key4, 4)] {
            if is_key_pressed(key) && k <= n_balls {
                balls = make_balls(n_balls, sw);
                for b in balls.iter_mut().take(k) {
                    b.theta = -0.9;
                }
            }
        }
        // audio + time controls on keys
        if is_key_pressed(KeyCode::M) {
            mute = !mute;
        }
        if is_key_pressed(KeyCode::Minus) {
            volume = (volume - 0.05).clamp(0.0, 1.0);
        }
        if is_key_pressed(KeyCode::Equal) {
            volume = (volume + 0.05).clamp(0.0, 1.0);
        }
        if is_key_pressed(KeyCode::Comma) {
            time_scale = (time_scale - 0.1).clamp(0.05, 2.0);
        }
        if is_key_pressed(KeyCode::Period) {
            time_scale = (time_scale + 0.1).clamp(0.05, 2.0);
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        let mouse = vec2(mouse_position().0, mouse_position().1);
        if is_mouse_button_pressed(MouseButton::Left) {
            active_slider = (0..SLIDERS.len()).find(|&i| slider_hit(i, mouse));
            if active_slider.is_none() {
                let mut best = None;
                let mut best_d = (BALL_R * 1.8).powi(2);
                for (i, b) in balls.iter().enumerate() {
                    let d = (b.pos() - mouse).length_squared();
                    if d < best_d {
                        best_d = d;
                        best = Some(i);
                    }
                }
                dragging = best;
            }
        }
        if let Some(i) = active_slider {
            let v = slider_value_from_mouse(i, mouse.x);
            match i {
                0 => damping = v,
                1 => room_size = v,
                _ => time_scale = v,
            }
        }
        if is_mouse_button_released(MouseButton::Left) {
            if active_slider == Some(1) {
                need_rebuild = true; // room size changed -> rebuild reverb
            }
            active_slider = None;
            dragging = None;
        }
        if let Some(i) = dragging {
            let dx = mouse.x - balls[i].pivot_x;
            let dy = (mouse.y - PIVOT_Y).max(10.0);
            balls[i].theta = dx.atan2(dy).clamp(-1.35, 1.35);
            balls[i].omega = 0.0;
        }

        // keep per-ball buffers sized to the (possibly rebuilt) ball list
        if trails.len() != balls.len() {
            trails = vec![Vec::new(); balls.len()];
            prev_theta = balls.iter().map(|b| b.theta).collect();
        }

        // ----- physics ----------------------------------------------------
        let real_dt = get_frame_time().min(1.0 / 30.0);
        let sim_dt = real_dt * if paused { 0.0 } else { time_scale };
        let dt = sim_dt / SUBSTEPS as f32;
        let mut frame_impact = 0.0_f32;
        let mut impact_at = vec2(sw * 0.5, sh * 0.5);

        for _ in 0..SUBSTEPS {
            for (i, b) in balls.iter_mut().enumerate() {
                if Some(i) == dragging {
                    continue;
                }
                let alpha = -(GRAVITY / STRING_L) * b.theta.sin();
                b.omega += alpha * dt;
                b.omega -= damping * b.omega * dt;
                b.theta += b.omega * dt;
            }
            for _ in 0..n_balls {
                let mut any = false;
                for i in 0..n_balls - 1 {
                    if Some(i) == dragging || Some(i + 1) == dragging {
                        continue;
                    }
                    let (pi, pj) = (balls[i].pos(), balls[i + 1].pos());
                    let closing = STRING_L * (balls[i].omega - balls[i + 1].omega);
                    if (pj - pi).length() <= 2.0 * BALL_R && closing > 1e-4 {
                        if closing > frame_impact {
                            frame_impact = closing;
                            impact_at = (pi + pj) * 0.5;
                        }
                        let tmp = balls[i].omega;
                        balls[i].omega = balls[i + 1].omega;
                        balls[i + 1].omega = tmp;
                        any = true;
                    }
                }
                if !any {
                    break;
                }
            }
        }

        // ----- clack sound + dust + screen shake on impact ----------------
        if frame_impact > MIN_AUDIBLE && dragging.is_none() {
            let t = frame_impact / SPEED_FULL;
            let idx = step_index(t, PITCH_STEPS) * PAN_STEPS + pan_index(impact_at.x, sw, n_balls);
            play_sound(
                &clack_bank[idx],
                PlaySoundParams { looped: false, volume: gain(0.12 + 0.88 * t.min(1.0), volume, mute) },
            );
            spawn_dust(&mut particles, impact_at, frame_impact);
            if frame_impact > SHAKE_MIN {
                let s = ((frame_impact - SHAKE_MIN) / SPEED_FULL).clamp(0.0, 1.0) * SHAKE_MAX_PX;
                shake = shake.max(s);
            }
        }

        // ----- whoosh as a ball rushes through the bottom -----------------
        let mut whoosh_speed = 0.0_f32;
        let mut whoosh_x = sw * 0.5;
        for (i, b) in balls.iter().enumerate() {
            if prev_theta[i] * b.theta < 0.0 {
                let sp = STRING_L * b.omega.abs();
                if sp > whoosh_speed {
                    whoosh_speed = sp;
                    whoosh_x = b.pos().x;
                }
            }
            prev_theta[i] = b.theta;
        }
        if whoosh_speed > WHOOSH_MIN && dragging.is_none() {
            let t = (whoosh_speed - WHOOSH_MIN) / (WHOOSH_FULL - WHOOSH_MIN);
            let idx = step_index(t, WHOOSH_STEPS) * PAN_STEPS + pan_index(whoosh_x, sw, n_balls);
            play_sound(
                &whoosh_bank[idx],
                PlaySoundParams { looped: false, volume: gain(0.10 + 0.45 * t.clamp(0.0, 1.0), volume, mute) },
            );
        }

        // ----- update trails, particles, shake ----------------------------
        for (i, b) in balls.iter().enumerate() {
            trails[i].push(b.pos());
            if trails[i].len() > TRAIL_LEN {
                trails[i].remove(0);
            }
        }
        update_particles(&mut particles, sim_dt);
        shake = (shake - SHAKE_DECAY * real_dt).max(0.0);

        // ----- render (under a shake-offset camera) -----------------------
        let off = if shake > 0.1 {
            vec2(gen_range(-shake, shake), gen_range(-shake, shake))
        } else {
            Vec2::ZERO
        };
        // NOTE: rendering through a Camera2D to the screen is Y-flipped versus
        // the default camera, so the y zoom is POSITIVE here to keep our
        // top-left / y-down world coordinates upright.
        let cam = Camera2D {
            zoom: vec2(2.0 / sw, 2.0 / sh),
            target: vec2(sw * 0.5 + off.x, sh * 0.5 + off.y),
            ..Default::default()
        };
        set_camera(&cam);
        let ui = Ui { damping, room: room_size, time_scale, volume, mute, paused };
        draw_scene(&balls, &trails, &particles, dragging, &ui, sw, sh);
        set_default_camera();

        // ----- persist settings when they change (and not mid-drag) -------
        let now = Config { damping, room: room_size, time_scale, volume, mute, n_balls };
        if now != last_saved && active_slider.is_none() {
            save_config(&now);
            last_saved = now;
        }

        next_frame().await;
    }
}

// ===== particles =========================================================

fn spawn_dust(particles: &mut Vec<Particle>, at: Vec2, impact: f32) {
    let strength = (impact / SPEED_FULL).clamp(0.0, 1.0);
    let count = (6.0 + strength * 18.0) as usize;
    for k in 0..count {
        let ang = gen_range(0.0, TAU);
        let speed = gen_range(30.0, 90.0) + impact * 0.22;
        let spark = strength > 0.45 && k < 5;
        particles.push(Particle {
            pos: at,
            vel: vec2(ang.cos() * speed, ang.sin() * speed - 30.0),
            life0: gen_range(0.30, 0.60),
            life: gen_range(0.30, 0.60),
            size: gen_range(1.5, 3.5) + if spark { 1.5 } else { 0.0 },
            spark,
        });
    }
    if particles.len() > 500 {
        let drop = particles.len() - 500;
        particles.drain(0..drop);
    }
}

fn update_particles(particles: &mut Vec<Particle>, dt: f32) {
    for p in particles.iter_mut() {
        p.vel.y += 900.0 * dt;
        p.vel *= 1.0 - (2.0 * dt).min(0.9);
        p.pos += p.vel * dt;
        p.life -= dt;
    }
    particles.retain(|p| p.life > 0.0);
}

// ===== rendering =========================================================

fn draw_scene(balls: &[Ball], trails: &[Vec<Vec2>], particles: &[Particle], dragging: Option<usize>, ui: &Ui, sw: f32, sh: f32) {
    clear_background(Color::new(0.08, 0.09, 0.12, 1.0));
    draw_rectangle(0.0, sh * 0.55, sw, sh * 0.45, Color::new(0.05, 0.06, 0.09, 1.0));

    let frame_col = Color::new(0.62, 0.65, 0.72, 1.0);
    let left = balls.first().map(|b| b.pivot_x).unwrap_or(sw * 0.5) - BALL_R - 30.0;
    let right = balls.last().map(|b| b.pivot_x).unwrap_or(sw * 0.5) + BALL_R + 30.0;
    draw_rectangle(left, PIVOT_Y - 14.0, right - left, 16.0, frame_col);
    for &x in &[left + 6.0, right - 18.0] {
        draw_rectangle(x, PIVOT_Y - 14.0, 12.0, sh * 0.78 - PIVOT_Y, frame_col);
        draw_rectangle(x - 24.0, sh * 0.78, 60.0, 12.0, frame_col);
    }

    for tr in trails {
        let len = tr.len();
        for k in 1..len {
            let a = k as f32 / len as f32;
            draw_line(tr[k - 1].x, tr[k - 1].y, tr[k].x, tr[k].y, 1.0 + a * 3.5, Color::new(0.45, 0.72, 1.0, a * 0.5));
        }
    }

    for (i, b) in balls.iter().enumerate() {
        let p = b.pos();
        draw_line(b.pivot_x, PIVOT_Y, p.x, p.y, 2.0, Color::new(0.8, 0.8, 0.85, 0.9));
        draw_circle(b.pivot_x, PIVOT_Y - 2.0, 4.0, Color::new(0.3, 0.32, 0.38, 1.0));
        let floor_y = sh * 0.78 + 12.0;
        draw_ellipse(p.x, floor_y, BALL_R * 0.9, BALL_R * 0.28, 0.0, Color::new(0.0, 0.0, 0.0, 0.25));

        let base = if Some(i) == dragging {
            Color::new(0.95, 0.82, 0.45, 1.0)
        } else {
            Color::new(0.78, 0.80, 0.86, 1.0)
        };
        draw_circle(p.x, p.y, BALL_R, Color::new(0.12, 0.13, 0.16, 1.0));
        draw_circle(p.x, p.y, BALL_R - 2.5, base);
        draw_circle(p.x, p.y, BALL_R * 0.62, Color::new(base.r, base.g, base.b, 0.55));
        draw_circle(p.x - BALL_R * 0.32, p.y - BALL_R * 0.34, BALL_R * 0.24, Color::new(1.0, 1.0, 1.0, 0.85));
    }

    for p in particles {
        let a = (p.life / p.life0).clamp(0.0, 1.0);
        let col = if p.spark {
            Color::new(1.0, 0.85, 0.45, a)
        } else {
            Color::new(0.85, 0.82, 0.75, a * 0.7)
        };
        draw_circle(p.pos.x, p.pos.y, p.size * a, col);
    }

    draw_readout(balls, sw);
    draw_sliders(ui);

    let dim = Color::new(0.7, 0.72, 0.8, 0.9);
    draw_text("Newton's Cradle", 18.0, 36.0, 30.0, Color::new(0.85, 0.87, 0.95, 1.0));
    draw_text(&format!("balls: {}   fps: {}", balls.len(), get_fps()), 18.0, 60.0, 18.0, Color::new(0.55, 0.58, 0.66, 1.0));
    draw_text(
        &format!(
            "vol {:.0}%{}   {}",
            ui.volume * 100.0,
            if ui.mute { " (muted)" } else { "" },
            if ui.paused { "PAUSED" } else { "" }
        ),
        18.0,
        sh - 42.0,
        18.0,
        dim,
    );
    draw_text(
        "drag ball/sliders  |  1-4 pull  |  R reset  |  Up/Down count  |  M mute  -/= vol  ,/. time  Space pause  |  Esc quit",
        18.0,
        sh - 18.0,
        18.0,
        dim,
    );
}

fn draw_sliders(ui: &Ui) {
    let vals = [ui.damping, ui.room, ui.time_scale];
    for (i, s) in SLIDERS.iter().enumerate() {
        draw_line(SLIDER_X, s.y, SLIDER_X + SLIDER_W, s.y, 4.0, Color::new(0.20, 0.22, 0.28, 1.0));
        let frac = ((vals[i] - s.lo) / (s.hi - s.lo)).clamp(0.0, 1.0);
        let kx = SLIDER_X + frac * SLIDER_W;
        draw_line(SLIDER_X, s.y, kx, s.y, 4.0, Color::new(0.45, 0.72, 1.0, 1.0));
        draw_circle(kx, s.y, 7.0, Color::new(0.9, 0.92, 0.98, 1.0));
        let txt = if i == 0 {
            format!("{}: {:.3}", s.label, vals[i])
        } else {
            format!("{}: {:.2}x", s.label, vals[i])
        };
        draw_text(&txt, SLIDER_X + SLIDER_W + 14.0, s.y + 5.0, 18.0, Color::new(0.72, 0.75, 0.82, 1.0));
    }
}

fn draw_readout(balls: &[Ball], sw: f32) {
    let mut ke = 0.0_f32;
    let mut pe = 0.0_f32;
    let mut px = 0.0_f32;
    for b in balls {
        let v = STRING_L * b.omega;
        ke += 0.5 * v * v;
        pe += GRAVITY * STRING_L * (1.0 - b.theta.cos());
        px += v * b.theta.cos();
    }
    let total = (ke + pe).max(1e-3);

    let x = sw - 250.0;
    let y = 84.0;
    let w = 226.0;
    draw_rectangle(x - 12.0, y - 26.0, w + 24.0, 104.0, Color::new(0.0, 0.0, 0.0, 0.30));
    draw_text("ENERGY (a.u.)", x, y - 6.0, 18.0, Color::new(0.75, 0.78, 0.85, 1.0));

    let ke_frac = ke / total;
    draw_rectangle(x, y, w, 14.0, Color::new(0.18, 0.20, 0.26, 1.0));
    draw_rectangle(x, y, w * ke_frac, 14.0, Color::new(0.40, 0.70, 1.00, 1.0));
    draw_rectangle(x + w * ke_frac, y, w * (1.0 - ke_frac), 14.0, Color::new(0.45, 0.85, 0.55, 1.0));
    draw_text(&format!("KE {:.0}   PE {:.0}", ke / 1000.0, pe / 1000.0), x, y + 34.0, 18.0, Color::new(0.7, 0.73, 0.8, 1.0));

    let my = y + 52.0;
    let cx = x + w * 0.5;
    draw_line(cx, my - 6.0, cx, my + 6.0, 1.0, Color::new(0.6, 0.6, 0.66, 0.8));
    let m = (px / 1500.0).clamp(-1.0, 1.0) * (w * 0.5);
    let mcol = Color::new(1.0, 0.78, 0.35, 1.0);
    if m >= 0.0 {
        draw_rectangle(cx, my - 4.0, m, 8.0, mcol);
    } else {
        draw_rectangle(cx + m, my - 4.0, -m, 8.0, mcol);
    }
    draw_text(&format!("momentum {:+.0}", px / 1000.0), x, my + 24.0, 18.0, Color::new(0.7, 0.73, 0.8, 1.0));
}
