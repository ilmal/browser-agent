import os
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys

def get_driver():

    print("Strating client...")

    base_url = "https://account.snapchat.com/accounts/v2/login"

    # Get resolution from environment variables
    screen_width = os.getenv('SCREEN_WIDTH', '1920')
    screen_height = os.getenv('SCREEN_HEIGHT', '1080')

    options = webdriver.ChromeOptions()
    options.add_argument(f"--window-size={screen_width},{screen_height}")
    # options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument('--start-maximized')
    options.add_argument('--no-sandbox')
    #options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(base_url)

    return driver