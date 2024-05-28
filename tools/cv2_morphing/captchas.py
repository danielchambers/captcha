import json
from random import choices
from string import ascii_uppercase, digits


def random_string(str_length, string_type="numbers"):
    return_value = ""
    if string_type == "numbers":
        return_value = "".join(choices(f"{digits}", k=str_length))
    else:
        return_value = "".join(choices(f"{ascii_uppercase}{digits}", k=str_length))
    return return_value


def site(name, random_str=False):
    links = json.loads(open("captcha_sites.json").read())

    if random_str:
        return f"{links[name]}{random_string(10)}"
    return links[name]


if __name__ == "__main__":
    main()
