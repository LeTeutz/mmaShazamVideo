# CALCULATE PARITY BIT
def cal_parity_bit(h, flag):
    ch, parity_list_local = 0, []

    for parity in range(0, (len(h))):
        ph = (2 ** ch)
        if ph == (parity + 1):
            startindex = ph - 1
            i, toxor = startindex, []

            while i < len(h):
                block = h[i:i + ph]
                toxor.extend(block)
                i += 2 * ph

            for z in range(1, len(toxor)):
                h[startindex] = h[startindex] ^ toxor[z]
            if flag:
                parity_list_local.append(h[parity])
            ch += 1
    if flag:
        return parity_list_local
    return h


# GENERATE HAMMING CODE
def generate_hamming_code(d):
    data = list(d)[::-1]
    c, j, r, h = 0, 0, 0, []

    while (len(d) + r + 1) > (pow(2, r)):
        r = r + 1

    for i in range(0, (r + len(data))):
        p = (2 ** c)

        if p == (i + 1):
            h.append(0)
            c = c + 1
        else:
            h.append(int(data[j]))
            j = j + 1

    h = cal_parity_bit(h, False)[::-1]
    return int(''.join(map(str, h)))


# DETECT ERROR IN RECEIVED HAMMING CODE
def detect_error_in_hamming_code(d):
    data = list(d)[::-1]
    c, error, h, parity_list, h_copy = 0, 0, [], [], []
    for k in range(0, len(data)):
        p = (2 ** c)
        h.append(int(data[k]))
        h_copy.append(data[k])
        if p == (k + 1):
            c = c + 1

    parity_list = cal_parity_bit(h, True)[::-1]
    error = sum(int(parity_list) * (2 ** i) for i, parity_list in enumerate(
        parity_list[::-1]))
    if error == 0:
        return 'There is no error in the hamming code received'
    elif error >= len(h_copy):
        return 'Error cannot be detected'
    h_copy[error - 1] = '1' if h_copy[error - 1] == '0' else '0'
    h_copy.reverse()
    correct_hamming_code = int(''.join(map(str, h_copy)))
    return 'Error is in,' + str(error) + ', bit\nAfter correction hamming ' \
                                         'code is:-\n' + str(
                                          correct_hamming_code)
