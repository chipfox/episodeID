"""Browser smoke. Run proposal_approval_test.py with EPID_BROWSER_PORT first."""
import os
import re

from playwright.sync_api import expect, sync_playwright

base = f"http://127.0.0.1:{os.environ['EPID_BROWSER_PORT']}"
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path=os.environ.get("EPID_CHROMIUM"))
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base + "/#/series/1")
    page.locator("button[data-f=review]").click()
    search = page.get_by_role("searchbox", name="Filter plan rows")
    search.evaluate("el => window.originalPlanSearch = el")
    search.press_sequentially("S01E01")
    assert search.evaluate("el => el === window.originalPlanSearch && el === document.activeElement")
    expect(page.locator(".plan-table tbody tr")).to_have_count(1)
    page.get_by_role("button", name="Clear filter").click()
    expect(search).to_have_value("")
    expect(page.get_by_role("button", name=re.compile("Review rename"))).to_have_count(1)
    original = page.request.get(base + "/api/series/1/plan").json()
    page.get_by_role("button", name="Review rename to S01E02").click()
    expect(page.get_by_role("button", name="Stage decision")).to_be_visible()
    page.get_by_role("button", name="Cancel", exact=True).click()
    assert page.request.get(base + "/api/series/1/plan").json() == original
    page.get_by_role("button", name="Review rename to S01E02").click()
    page.get_by_role("button", name="Stage decision").click()
    expect(page.get_by_role("button", name="Apply 1 change…")).to_be_enabled()
    assert not errors, errors
    browser.close()
print("PASS (search keeps focus, weak candidates lack approval, cancel is read-only, approval stages)")
