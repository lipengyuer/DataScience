freq_brand_id_list = "1446  3738  1214  5434  1360  2276  5376  82  8235  4705  3969  1662  4073  7069  2104  376  7749  3650  3700  6143  6065  4874  6585  856  6938  4509  4290  3929  1859  4953  6742  1573  4594  3535  99  1866  6762"
freq_brand_id_list = freq_brand_id_list.split('  ')
data = {}
for i in range(len(freq_brand_id_list)):
    data[freq_brand_id_list[i]] = i
print(data)