#include <TFT_eSPI.h>
#include <SPI.h>

TFT_eSPI tft = TFT_eSPI();
TFT_eSprite spr = TFT_eSprite(&tft);

const int CX = 120;  // center X
const int CY = 120;  // center Y

// === EYE STYLE (change via serial: EYE1, EYE2, EYE3, EYE4, EYE5) ===
int eyeStyle = 1;

// Common state
float pupilX = 0, pupilY = 0;
float targetX = 0, targetY = 0;
float glowPhase = 0;
String currentMode = "IDLE";

// ---- Blink state machine ----
#define NOBLINK   0
#define ENBLINK   1
#define HOLDBLINK 2
#define DEBLINK   3

struct BlinkState {
  uint8_t  state;
  uint32_t startTime;
  uint32_t closeDur, holdDur, openDur;
  float    amount;     // 0=open, 1=closed
  uint32_t nextBlink;
} blink = {NOBLINK, 0, 150, 50, 250, 0.0, 3000};

float smoothstep(float t) {
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  return t * t * (3.0f - 2.0f * t);
}

void triggerBlink() {
  blink.state = ENBLINK;
  blink.startTime = millis();
  blink.closeDur = 150 + random(50);
  blink.holdDur  = 50 + random(50);
  blink.openDur  = 250 + random(150);
}

void updateBlink() {
  uint32_t now = millis();
  if (blink.state == NOBLINK && now >= blink.nextBlink) triggerBlink();
  uint32_t elapsed = now - blink.startTime;
  float t;
  switch (blink.state) {
    case NOBLINK: blink.amount = 0.0; break;
    case ENBLINK:
      t = (float)elapsed / blink.closeDur;
      if (t >= 1.0) { blink.amount = 1.0; blink.state = HOLDBLINK; blink.startTime = now; }
      else blink.amount = smoothstep(t);
      break;
    case HOLDBLINK:
      blink.amount = 1.0;
      if (elapsed >= blink.holdDur) { blink.state = DEBLINK; blink.startTime = now; }
      break;
    case DEBLINK:
      t = (float)elapsed / blink.openDur;
      if (t >= 1.0) { blink.amount = 0.0; blink.state = NOBLINK; blink.nextBlink = now + 2000 + random(4000); }
      else blink.amount = 1.0 - smoothstep(t);
      break;
  }
}

// ---- Utility ----
uint16_t dimColor(uint16_t color, float f) {
  if (f > 2.0) f = 2.0; if (f < 0.0) f = 0.0;
  uint8_t r = ((color >> 11) & 0x1F);
  uint8_t g = ((color >> 5) & 0x3F);
  uint8_t b = (color & 0x1F);
  return (constrain((int)(r*f),0,31)<<11) | (constrain((int)(g*f),0,63)<<5) | constrain((int)(b*f),0,31);
}

uint16_t rgb(uint8_t r, uint8_t g, uint8_t b) {
  return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

// ---- Flat metallic eyelids (shared by all styles) ----
void drawEyelids(float amt) {
  if (amt <= 0.01) return;
  int lidW = 140;  // half-width
  int curve = 10;  // very flat

  // Upper lid
  int uOpen = CY - 110;
  int uClose = CY + 25;
  int uEdge = uOpen + (int)((uClose - uOpen) * amt);
  for (int y = 0; y < min(uEdge + curve + 1, 240); y++) {
    float d = (float)(y - uEdge) / curve;
    if (d > 1.0) break;
    float wf = (d <= 0) ? 1.0 : max(1.0f - d*d, 0.0f);
    int hw = (int)(lidW * wf);
    int x1 = max(CX - hw, 0), x2 = min(CX + hw, 239);
    if (x2 <= x1) continue;
    uint16_t col = (y < uEdge - 3) ? rgb(50,50,55) : (y < uEdge) ? rgb(90,90,95) : rgb(130,130,140);
    spr.drawFastHLine(x1, y, x2 - x1, col);
  }

  // Lower lid (40%)
  float la = amt * 0.4;
  int lOpen = CY + 110;
  int lClose = CY - 15;
  int lEdge = lOpen + (int)((lClose - lOpen) * la);
  for (int y = max(lEdge - curve - 1, 0); y < 240; y++) {
    float d = (float)(lEdge - y) / curve;
    if (d > 1.0) continue;
    float wf = (d <= 0) ? 1.0 : max(1.0f - d*d, 0.0f);
    int hw = (int)(lidW * wf);
    int x1 = max(CX - hw, 0), x2 = min(CX + hw, 239);
    if (x2 <= x1) continue;
    uint16_t col = (y > lEdge + 3) ? rgb(50,50,55) : (y > lEdge) ? rgb(90,90,95) : rgb(130,130,140);
    spr.drawFastHLine(x1, y, x2 - x1, col);
  }
}

// ================================================================
// EYE STYLE 1: Classic T-800 — red iris, radial lines, dark sclera
// ================================================================
void drawEye1(float ox, float oy, float intensity) {
  int cx = CX + (int)ox, cy = CY + (int)oy;
  spr.fillSprite(TFT_BLACK);

  // Dark red sclera
  spr.fillCircle(CX, CY, 100, dimColor(0x3800, intensity * 0.6));
  // Red iris
  spr.fillCircle(cx, cy, 55, dimColor(0xF800, intensity));
  // Iris ring
  for (int r = 55; r > 50; r--) spr.drawCircle(cx, cy, r, dimColor(0xF800, intensity * 1.3));
  // Radial lines
  for (int a = 0; a < 360; a += 15) {
    float rad = a * PI / 180.0;
    spr.drawLine(cx+cos(rad)*28, cy+sin(rad)*28, cx+cos(rad)*49, cy+sin(rad)*49, dimColor(0xA000, intensity));
  }
  // Pupil
  spr.fillCircle(cx, cy, 25, TFT_BLACK);
  // Highlights
  spr.fillCircle(cx-8, cy-8, 4, dimColor(0xFB20, intensity));
  spr.fillCircle(cx+5, cy-5, 2, dimColor(0xF800, intensity*0.5));

  drawEyelids(blink.amount);
  spr.pushSprite(0, 0);
}

// ================================================================
// EYE STYLE 2: Minimal glowing ring — no sclera, just a bright
//              red ring with a dark center. Clean, cybernetic.
// ================================================================
void drawEye2(float ox, float oy, float intensity) {
  int cx = CX + (int)ox, cy = CY + (int)oy;
  spr.fillSprite(TFT_BLACK);

  // Outer glow halo
  for (int r = 75; r > 60; r--) {
    float f = (float)(r - 60) / 15.0;
    spr.drawCircle(cx, cy, r, dimColor(0xF800, intensity * (1.0 - f) * 0.3));
  }
  // Main bright ring
  for (int r = 60; r > 45; r--) {
    spr.drawCircle(cx, cy, r, dimColor(0xF800, intensity * 1.2));
  }
  // Inner dark
  spr.fillCircle(cx, cy, 45, TFT_BLACK);
  // Tiny bright pupil dot
  spr.fillCircle(cx, cy, 6, dimColor(0xF800, intensity * 0.8));

  drawEyelids(blink.amount);
  spr.pushSprite(0, 0);
}

// ================================================================
// EYE STYLE 3: Realistic human-ish eye — white sclera, amber/orange
//              iris with detailed texture, black pupil, veins
// ================================================================
void drawEye3(float ox, float oy, float intensity) {
  int cx = CX + (int)ox, cy = CY + (int)oy;
  spr.fillSprite(TFT_BLACK);

  // White sclera
  uint16_t scleraCol = dimColor(rgb(220, 215, 210), intensity * 0.9);
  spr.fillCircle(CX, CY, 100, scleraCol);

  // Subtle veins
  uint16_t veinCol = dimColor(rgb(180, 100, 100), intensity * 0.4);
  spr.drawLine(CX-95, CY-10, CX-55, CY-5, veinCol);
  spr.drawLine(CX-90, CY+15, CX-50, CY+8, veinCol);
  spr.drawLine(CX+95, CY-8, CX+55, CY-3, veinCol);
  spr.drawLine(CX+88, CY+12, CX+52, CY+6, veinCol);

  // Amber/orange iris with gradient
  for (int r = 50; r > 0; r--) {
    float t = (float)r / 50.0;
    uint16_t irisCol;
    if (t > 0.7) irisCol = dimColor(rgb(140, 80, 20), intensity);      // dark outer
    else if (t > 0.4) irisCol = dimColor(rgb(200, 120, 30), intensity); // mid amber
    else irisCol = dimColor(rgb(180, 100, 20), intensity);              // inner
    spr.drawCircle(cx, cy, r, irisCol);
  }
  // Iris outer ring
  for (int r = 52; r > 49; r--) spr.drawCircle(cx, cy, r, dimColor(rgb(60, 40, 20), intensity));
  // Radial iris fibers
  for (int a = 0; a < 360; a += 10) {
    float rad = a * PI / 180.0;
    uint16_t fCol = dimColor(rgb(160, 90, 20), intensity * 0.7);
    spr.drawLine(cx+cos(rad)*18, cy+sin(rad)*18, cx+cos(rad)*46, cy+sin(rad)*46, fCol);
  }
  // Pupil
  spr.fillCircle(cx, cy, 18, TFT_BLACK);
  // Highlights
  spr.fillCircle(cx-10, cy-10, 6, dimColor(rgb(255,255,255), intensity * 0.9));
  spr.fillCircle(cx+6, cy-6, 3, dimColor(rgb(255,255,255), intensity * 0.5));

  drawEyelids(blink.amount);
  spr.pushSprite(0, 0);
}

// ================================================================
// EYE STYLE 4: Sci-fi HUD eye — concentric rings, scanning lines,
//              digital/holographic look with cyan/blue tones
// ================================================================
void drawEye4(float ox, float oy, float intensity) {
  int cx = CX + (int)ox, cy = CY + (int)oy;
  spr.fillSprite(TFT_BLACK);

  uint16_t cyan = rgb(0, 200, 255);
  uint16_t darkCyan = rgb(0, 60, 80);

  // Concentric rings
  for (int r = 100; r > 10; r -= 8) {
    spr.drawCircle(cx, cy, r, dimColor(darkCyan, intensity * 0.6));
  }
  // Brighter inner rings
  for (int r = 55; r > 30; r -= 4) {
    spr.drawCircle(cx, cy, r, dimColor(cyan, intensity * 0.8));
  }
  // Cross-hairs
  spr.drawFastHLine(cx - 95, cy, 190, dimColor(cyan, intensity * 0.3));
  spr.drawFastVLine(cx, cy - 95, 190, dimColor(cyan, intensity * 0.3));
  // Diagonal lines
  spr.drawLine(cx-70, cy-70, cx+70, cy+70, dimColor(darkCyan, intensity*0.2));
  spr.drawLine(cx+70, cy-70, cx-70, cy+70, dimColor(darkCyan, intensity*0.2));

  // Animated scan line
  int scanY = cy + (int)(sin(glowPhase * 2) * 60);
  spr.drawFastHLine(cx - 80, scanY, 160, dimColor(cyan, intensity * 0.6));
  spr.drawFastHLine(cx - 60, scanY+1, 120, dimColor(cyan, intensity * 0.3));

  // Center iris
  spr.fillCircle(cx, cy, 28, dimColor(darkCyan, intensity));
  for (int r = 28; r > 25; r--) spr.drawCircle(cx, cy, r, dimColor(cyan, intensity * 1.3));
  // Pupil
  spr.fillCircle(cx, cy, 12, TFT_BLACK);
  spr.fillCircle(cx, cy, 3, dimColor(cyan, intensity * 0.5));

  drawEyelids(blink.amount);
  spr.pushSprite(0, 0);
}

// ================================================================
// EYE STYLE 5: Demon/dragon eye — vertical slit pupil, yellow-green
//              iris, fiery orange outer, reptilian texture
// ================================================================
void drawEye5(float ox, float oy, float intensity) {
  int cx = CX + (int)ox, cy = CY + (int)oy;
  spr.fillSprite(TFT_BLACK);

  // Dark outer ring
  spr.fillCircle(CX, CY, 100, dimColor(rgb(40, 15, 0), intensity));

  // Fiery orange-red outer iris
  for (int r = 85; r > 55; r--) {
    float t = (float)(r - 55) / 30.0;
    uint16_t c = dimColor(rgb(200, 60 + (int)(t*60), 0), intensity);
    spr.drawCircle(cx, cy, r, c);
  }
  // Yellow-green inner iris
  for (int r = 55; r > 15; r--) {
    float t = (float)(r - 15) / 40.0;
    uint16_t c = dimColor(rgb(180 + (int)(t*40), 200 - (int)(t*80), 0), intensity);
    spr.drawCircle(cx, cy, r, c);
  }
  // Iris ring
  for (int r = 87; r > 84; r--) spr.drawCircle(cx, cy, r, dimColor(rgb(80, 30, 0), intensity));

  // Reptilian radial texture
  for (int a = 0; a < 360; a += 8) {
    float rad = a * PI / 180.0;
    spr.drawLine(cx+cos(rad)*18, cy+sin(rad)*18, cx+cos(rad)*82, cy+sin(rad)*82,
                 dimColor(rgb(100, 40, 0), intensity * 0.5));
  }

  // Vertical slit pupil
  int slitW = 8;  // half width at center
  for (int y = cy - 65; y <= cy + 65; y++) {
    float dy = (float)abs(y - cy) / 65.0;
    int w = (int)(slitW * (1.0 - dy * dy));
    if (w < 1) w = 1;
    spr.drawFastHLine(cx - w, y, w * 2, TFT_BLACK);
  }

  // Tiny bright highlight
  spr.fillCircle(cx - 12, cy - 20, 4, dimColor(rgb(255, 255, 200), intensity * 0.7));

  drawEyelids(blink.amount);
  spr.pushSprite(0, 0);
}

// ================================================================
// Dispatch: draw current eye style
// ================================================================
void drawCurrentEye(float ox, float oy, float intensity) {
  switch (eyeStyle) {
    case 1: drawEye1(ox, oy, intensity); break;
    case 2: drawEye2(ox, oy, intensity); break;
    case 3: drawEye3(ox, oy, intensity); break;
    case 4: drawEye4(ox, oy, intensity); break;
    case 5: drawEye5(ox, oy, intensity); break;
    default: drawEye1(ox, oy, intensity); break;
  }
}

// ---- Animation modes ----
void updateIdle() {
  glowPhase += 0.05;
  float intensity = 1.0 + 0.3 * sin(glowPhase);
  if (random(100) < 3) { targetX = random(-15, 16); targetY = random(-10, 11); }
  pupilX += (targetX - pupilX) * 0.1;
  pupilY += (targetY - pupilY) * 0.1;
  drawCurrentEye(pupilX, pupilY, intensity);
}

void updateScanning() {
  glowPhase += 0.08;
  drawCurrentEye(sin(glowPhase) * 30, 0, 1.2 + 0.4 * sin(glowPhase * 3));
}

void updateAlert() {
  glowPhase += 0.15;
  drawCurrentEye(pupilX, pupilY, 1.0 + 0.8 * abs(sin(glowPhase)));
}

void updateListening() {
  glowPhase += 0.03;
  pupilX *= 0.9; pupilY *= 0.9;
  drawCurrentEye(pupilX, pupilY, 0.8 + 0.2 * sin(glowPhase));
}

// ---- Setup & Loop ----
void setup() {
  Serial.begin(115200);
  Serial.println("T-800 Eye System Booting...");

  tft.init();
  tft.setRotation(0);
  tft.fillScreen(TFT_BLACK);

  spr.setColorDepth(8);
  void *p = spr.createSprite(240, 240);
  Serial.println(p ? "[OK] Sprite allocated." : "[ERROR] Sprite FAILED!");
  Serial.print("[MEM] Free heap: "); Serial.println(ESP.getFreeHeap());

  blink.nextBlink = millis() + 2000 + random(3000);

  Serial.println("=== Eye Styles ===");
  Serial.println("EYE1 — Classic T-800 (red iris, dark sclera)");
  Serial.println("EYE2 — Minimal Ring (cybernetic glow ring)");
  Serial.println("EYE3 — Realistic Human (white sclera, amber iris)");
  Serial.println("EYE4 — Sci-fi HUD (cyan rings, scan lines)");
  Serial.println("EYE5 — Dragon/Demon (vertical slit, yellow-green)");
  Serial.println("=== Commands: IDLE, SCANNING, ALERT, LISTENING, BLINK ===");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd == "BLINK") { triggerBlink(); }
    else if (cmd == "EYE1") { eyeStyle = 1; Serial.println("Style: Classic T-800"); }
    else if (cmd == "EYE2") { eyeStyle = 2; Serial.println("Style: Minimal Ring"); }
    else if (cmd == "EYE3") { eyeStyle = 3; Serial.println("Style: Realistic Human"); }
    else if (cmd == "EYE4") { eyeStyle = 4; Serial.println("Style: Sci-fi HUD"); }
    else if (cmd == "EYE5") { eyeStyle = 5; Serial.println("Style: Dragon/Demon"); }
    else if (cmd.length() > 0) {
      currentMode = cmd;
      Serial.println("Mode: " + currentMode);
    }
  }

  updateBlink();

  if (currentMode == "IDLE") updateIdle();
  else if (currentMode == "SCANNING") updateScanning();
  else if (currentMode == "ALERT") updateAlert();
  else if (currentMode == "LISTENING") updateListening();
  else if (currentMode == "LOOK_LEFT") {
    targetX = -30; pupilX += (targetX - pupilX) * 0.1;
    drawCurrentEye(pupilX, pupilY, 1.0);
  } else if (currentMode == "LOOK_RIGHT") {
    targetX = 30; pupilX += (targetX - pupilX) * 0.1;
    drawCurrentEye(pupilX, pupilY, 1.0);
  } else updateIdle();

  delay(30);
}
