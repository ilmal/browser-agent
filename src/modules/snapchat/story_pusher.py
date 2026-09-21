import os
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys

def get_element(driver, input, search_type="XPATH"):
    try:
        DELAY = 60
        if search_type == "XPATH":
            return WebDriverWait(driver, DELAY).until(EC.presence_of_element_located(
                (By.XPATH, str(input))))
        if search_type == "CLASS_NAME":
            return WebDriverWait(driver, DELAY).until(EC.visibility_of_all_elements_located(
                (By.CLASS_NAME, str(input))))
    except TimeoutException:
        print("Loading took too much time!")
        driver.quit()

def get_driver():

    print("sending snap")

    base_url = "https://my.snapchat.com/"

    dirname = os.path.dirname(__file__)
    driver_user_path = dirname.replace("modules", "driver_user")

    # Get resolution from environment variables
    screen_width = os.getenv('SCREEN_WIDTH', '1920')
    screen_height = os.getenv('SCREEN_HEIGHT', '1080')

    """

Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36

    """

    options = webdriver.ChromeOptions()
    options.add_argument(f"--window-size={screen_width},{screen_height}")
    #options.add_argument("--headless=new")
    #options.add_argument("--disable-gpu")
    #options.add_argument('--start-maximized')
    options.add_argument('--no-sandbox')
    #options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
    #options.add_argument(str('--user-data-dir=' + driver_user_path)) # use a custom profile

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(base_url)

    return driver

def post_story():
    driver = get_driver()

    #Press login
    button = get_element(driver, "/html/body/div/main/div[2]/div[2]/div[2]/div/div[2]/div/div/button")
    button.click()

    # video drop in
    time.sleep(1)
    input  = get_element(driver, "/html/body/div/main/div[2]/div[2]/div[1]/div/div/input")
    input.send_keys(os.getcwd()+"/src/images/story_image.jpg")

    # select public story
    button = get_element(driver, "/html/body/div/main/div[2]/div[2]/div[2]/div[5]/div[2]/div/div/div[2]/div/div/div/div[2]/div/button")
    button.click()

    # send story
    time.sleep(1)
    button = get_element(driver, "/html/body/div/main/div[2]/div[2]/div[2]/div[6]/div/div[2]/div/div[2]/div/button")
    ActionChains(driver).move_to_element(button).perform()
    button.click()

    driver.quit()

if __name__ == "__main__":
    post_story()