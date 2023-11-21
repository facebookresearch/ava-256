







tmp = open('32ids.txt', 'rt').readlines()

carray=str("")
sidarray = str("")
for e in tmp:
    toks = e.split("-")
    mcd = toks[0]
    mct = toks[1]
    msid = toks[2].rstrip()

    crypto=f"AIRSTORE_CODEC_AVATAR_{mcd}_{mct}_{msid.upper()}_FOV_FULL_ORDER_BY_FRAME_FLASHARRAY_CRYPTO"

    carray += crypto + " "
    sidarray += e.rstrip() + " "

    print(crypto)


print(" C array ")
print(carray)
print(" SID array ")

print(sidarray)
