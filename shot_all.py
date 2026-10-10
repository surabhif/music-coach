"""Screenshots + browser e2e: studio section select, fake-mic recording -> analysis, notes, Indian, progress."""
import json, sqlite3
from playwright.sync_api import sync_playwright
B = "http://localhost:7860"
with sync_playwright() as p:
    br = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                 "--use-file-for-fake-audio-capture=/tmp/fake_mic.wav", "--autoplay-policy=no-user-gesture-required"])
    ctx = br.new_context(viewport={"width": 1360, "height": 900}); ctx.grant_permissions(["microphone"], origin=B)
    pg = ctx.new_page(); pg.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const st=document.createElement('style');st.textContent='.nav{position:relative!important}';document.head.appendChild(st)})"); pg.on("console", lambda m: print("console:", m.text) if m.type == "error" else None)
    # ---- studio: pick the last auto section, show loop selection
    pg.goto(B + "/studio?expert=demo_expert.wav"); pg.wait_for_selector("#chips .chip", timeout=120000)
    pg.locator("#chips .chip").last.click(); pg.fill("[name=singer]", "Demo Singer"); pg.wait_for_timeout(400)
    print("selection:", pg.inner_text("#selt"))
    pg.screenshot(path="screens/section.png", full_page=True)
    # ---- record with count-in (fake mic plays a take of that section)
    pg.select_option("#beats", "2"); pg.click("#rec"); pg.wait_for_timeout(2600)
    pg.screenshot(path="screens/record.png", full_page=True)
    print("status during:", pg.inner_text("#rstat"))
    pg.wait_for_function("document.getElementById('take').style.display==='flex'", timeout=30000)
    print("status after:", pg.inner_text("#rstat"))
    pg.click("#go"); pg.wait_for_url("**/result/**", timeout=300000); pg.wait_for_timeout(3000)
    rec_url = pg.url; print("recorded result:", rec_url, pg.inner_text(".ring .num"), pg.inner_text(".pill"))
    # ---- notes view on a result with issues (detuned take)
    rid = sqlite3.connect("vocalmatch.db").execute("select rid from attempts where id=4").fetchone()[0]
    pg.goto(f"{B}/result/{rid}"); pg.wait_for_timeout(3000)
    pg.locator("#roll").scroll_into_view_if_needed(); pg.wait_for_timeout(500)
    box = pg.locator("#roll").bounding_box()
    pg.screenshot(path="screens/notes.png", clip=dict(x=0, y=max(0, box["y"] - 150), width=1360, height=560))
    pg.locator(".wl li[data-t]").first.click(); pg.wait_for_timeout(1200)
    print("seek -> user player t=", pg.evaluate("P.pu.currentTime.toFixed(2)"), "expert t=", pg.evaluate("P.pe.currentTime.toFixed(2)"))
    pg.screenshot(path="screens/results_full.png", full_page=True)
    # ---- Indian classical result (off-raga take)
    rid = sqlite3.connect("vocalmatch.db").execute("select rid from attempts where id=2").fetchone()[0]
    pg.goto(f"{B}/result/{rid}"); pg.wait_for_timeout(3000); pg.screenshot(path="screens/indian.png", full_page=True)
    print("indian banner:", pg.inner_text(".banner")[:160])
    # ---- home with Indian options open
    pg.goto(B + "/"); pg.click("label[for=m-i]"); pg.wait_for_timeout(300); pg.screenshot(path="screens/home.png", full_page=True)
    # ---- progress
    pg.goto(B + "/history?singer=Demo%20Singer&song=demo_expert.wav"); pg.wait_for_timeout(3000); pg.screenshot(path="screens/progress.png", full_page=True)
    br.close()
