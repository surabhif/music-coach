from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={"width":1360,"height":900})
    pg.goto("http://localhost:7860/"); pg.set_input_files("#user","test_user_transposed_down2.wav")
    assert "Key-tolerant" in pg.content(); pg.click("#go")
    pg.wait_for_url("**/result/**",timeout=300000); pg.wait_for_timeout(3500)
    pg.screenshot(path="screens/results_key.png",full_page=True)
    print(pg.url, pg.inner_text("text=You sang"))
