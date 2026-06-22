from __future__ import annotations

import getpass
import os

from dhan_auth import generate_dhan_access_token


def main() -> None:
    client_id = str(os.getenv("DHAN_CLIENT_ID", "")).strip() or input("Dhan Client ID: ").strip()
    pin = str(os.getenv("DHAN_PIN", "")).strip() or getpass.getpass("Dhan PIN: ").strip()
    totp_code = input("Dhan TOTP Code (press Enter to use secret): ").strip()
    totp_secret = ""
    if not totp_code:
        totp_secret = getpass.getpass("Dhan TOTP Secret: ").strip()
    token = generate_dhan_access_token(
        client_id,
        pin,
        totp_secret=totp_secret,
        totp_code=totp_code,
    )
    print("\nDhan access token:\n")
    print(token)


if __name__ == "__main__":
    main()
