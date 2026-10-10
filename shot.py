import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1360,"height":900})
    pg.goto("http://localhost:7860/"); pg.wait_for_timeout(1500)
    pg.set_input_files("#user","test_user_slightly_off.wav"); pg.wait_for_timeout(300)
    pg.screenshot(path="screens/home.png",full_page=True)
    pg.click("#go"); pg.wait_for_timeout(1500); pg.screenshot(path="screens/loading.png")
    pg.wait_for_url("**/result/**",timeout=300000); pg.wait_for_timeout(3500)
    pg.screenshot(path="screens/results.png",full_page=True); print(pg.url)
    m=b.new_page(viewport={"width":390,"height":844}); m.goto(pg.url); m.wait_for_timeout(2500); m.screenshot(path="screens/results_mobile.png",full_page=True)
