import time
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys

from modules.get_element import get_element

def check_for_messages(driver):
    user_bodys = get_element(driver, "O4POs", "CLASS_NAME")

    print(user_bodys)

    if not user_bodys:
        return None

    for index, user_body in enumerate(user_bodys):
        name = user_body.find_element(By.XPATH, "./div[2]/span").text
        message = user_body.find_element(By.XPATH, "./div[2]/div").text

        if "New Chat" in message:
            print(name)
            return [name, user_bodys[index]]

def read_messages(driver, element):
    element.click()
    ActionChains(driver).send_keys(Keys.CONTROL + "R")
    time.sleep(.5)

    chats = driver.find_elements(By.CLASS_NAME, "T1yt2")
    
    latest_chat = chats[-1].text.split("\n")[-1]

    # return to main page:
    button = get_element(driver, "/html/body/main/div[1]/div[2]/div/div/div/div[1]/div[1]/button")
    button.click()

    return latest_chat


def snap_listener(driver):

    if not driver:
        print("Driver is None")
        return False, False

    new_messages = check_for_messages(driver) # returns [user, element]
    if new_messages:
        latest_chat = read_messages(driver, new_messages[1])

        print("RETURNNIG: ", new_messages[0], latest_chat)

        return new_messages[0], latest_chat
    
    print("No new messages")
    return False, False 
            

if __name__ == "__main__":
    pass