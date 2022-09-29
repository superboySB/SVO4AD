import datetime

NUM_COLS = 25

GLB_vehID_colidx = 0
GLB_frmID_colidx = 1
GLB_totfrm_colidx = 2
GLB_glbtime_colidx = 3
GLB_locx_colidx = 4
GLB_locy_colidx = 5
GLB_glbx_colidx = 6
GLB_glby_colidx = 7
GLB_vehlen_colidx = 8
GLB_vehwid_colidx = 9
GLB_vehcls_colidx = 10
GLB_vehspd_colidx = 11
GLB_vehacc_colidx = 12
GLB_laneID_colidx = 13
GLB_Ozone_colidx = 14
GLB_Dzone_colidx = 15
GLB_interID_colidx = 16
GLB_sectID_colidx = 17
GLB_dirc_colidx = 18
GLB_mov_colidx = 19
GLB_pred_colidx = 20
GLB_follow_colidx = 21
GLB_shead_colidx = 22
GLB_thead_colidx = 23
GLB_loc_colidx = 24

GLB_DEBUG = False
GLB_ROUNDING_100MS = -2
GLB_UNIXTIME_GAP = 100
GLB_TIME_THRES = 10000
GLB_DETECT_TOL = 0.9

timezone_dict = dict()
timezone_dict['i-80'] = 'America/Los_Angeles'
timezone_dict['us-101'] = 'America/Los_Angeles'
timezone_dict['lankershim'] = 'America/Los_Angeles'

GLB_LANE_CONSIDERED = dict()
GLB_LANE_CONSIDERED['i-80'] = [1, 2, 3, 4, 5, 6]
GLB_LANE_CONSIDERED['us-101'] = [1, 2, 3, 4, 5]
GLB_LANE_CONSIDERED['lankershim'] = [0, 1, 2, 3, 4, 11, 12, 31, 101]

GLOB_IMPUTE_K_SWEEP = [1, 3, 5, 10, 13, 15, 18, 20]

vehicles_us101 = [515, 2127, 1033, 1890, 744, 496, 946, 1779, 2780, 1212, 2182, 5, 1282, 1123, 686, 709, 485, 182, 1320,
                  2938, 143, 1794, 1537, 1476, 1323, 1262, 804, 346, 1573, 707, 131, 824, 2612, 1030, 694, 1220, 1283,
                  442, 449, 478, 2240, 718, 717, 1610, 52, 141, 1171, 1584, 481, 547, 1679, 73, 1354, 762, 1845, 1197,
                  1693, 499, 376, 1951, 610, 378, 645, 2958, 1806, 1396, 1789, 1657, 554, 1357, 2561, 2138, 749, 1441,
                  1885, 1580, 2049, 388, 1896, 292, 1583, 165, 1287, 2470, 653, 2581, 2063, 1416, 1884, 431, 1652, 6,
                  1747, 2053, 2393, 2196, 2515, 756, 1499, 2031, 616, 313, 331, 2467, 290, 2169, 1205, 2523, 1365, 2159,
                  1570, 2365, 41, 256, 1748, 1506, 1841, 2598, 2972, 1383, 603, 401, 77, 2861, 2832, 2605, 650, 2282,
                  1696, 1489, 2115, 1341, 2000, 14, 368, 567, 1223, 2118, 2297, 186, 479, 1064, 994, 583, 1670, 623,
                  566, 1112, 2841, 1271, 1056, 1547, 260, 818, 779, 2871, 1562, 1599, 2669, 361, 2542, 1368, 809, 1232,
                  1566, 2399, 534, 1077, 724, 2317, 26, 2037, 2189, 1267, 2133, 2345, 969, 1116, 2916, 2715, 1435, 921,
                  1507, 62, 1706, 1586, 2921, 1645, 527, 1857, 2691, 1731, 308, 1790, 1714, 795, 1445, 1460, 386, 454,
                  769, 2860, 1925, 1920, 2649, 275, 1179, 2004, 422, 2982, 514, 1312, 1333, 735, 1773, 1080, 232, 1937,
                  1934, 574, 72, 1863, 1525, 1517, 2507, 190, 21, 112, 1298, 2032, 1204, 1681, 1226, 118, 1672, 2880,
                  1643, 475, 1940, 838, 1612, 1604, 1334, 1703, 1976, 2313, 1942, 263, 1172, 1039, 2168, 1746, 850, 625,
                  1687, 2931, 2191, 1595, 113, 1138, 1898, 1373, 1981, 1647, 345, 2306, 219, 1947, 1452, 627, 2510, 998,
                  1522, 1484, 231, 1035, 2034, 1840, 109, 1388, 1616, 776, 2062, 1191, 588, 886, 261, 783, 2704, 1637,
                  1935, 1455, 589, 712, 207, 1613, 1279, 2319, 1168, 1192, 2374, 87, 1044, 1995, 416, 152, 1555, 1144,
                  1185, 1427, 1128, 2848, 2177, 248, 1514, 1142, 1539, 1337, 340, 1769, 2360, 693, 1046, 50, 2804, 1304,
                  816, 1771, 2386, 2128, 1173, 498, 1093, 536, 1966, 149, 1069, 1495, 2795, 906, 1684, 153, 303, 1908,
                  2160, 1582, 548, 1558, 1397, 1292, 2524, 1339, 2092, 1755, 751, 1715, 1808, 1353, 1802, 1778, 1552,
                  1490, 2214, 846, 2192, 877, 147, 1177, 2002, 1783, 1500, 543, 91, 1743, 1600, 728, 1770, 2764, 1248,
                  1923, 253, 1081, 82, 287, 1551, 1820, 1541, 1835, 1067, 1055, 329, 328, 334, 1470, 1302, 2616, 2349,
                  2823, 2224, 1870, 1642, 467, 753, 1417, 1815, 2983, 704, 1401, 1716, 888, 679, 1319, 556, 1201, 2727,
                  927, 1550, 2301, 1818, 2678, 1463, 1624, 1199, 607, 2552, 18, 699, 1422, 183, 447, 430, 1300, 2597,
                  986, 1257, 1677, 591, 205, 2163, 2088, 895, 2325, 121, 1159, 2441, 2740, 1824, 1249, 931, 2443, 985,
                  1557, 1626, 1694, 967, 282, 150, 1720, 236, 1690, 2322, 381, 123, 695, 1536, 1178, 2232, 1295, 1356,
                  2132, 1812, 841, 2613, 2272, 2353, 2820, 2302, 1203, 1698, 2535, 2411, 1727, 680, 1709, 297, 2271,
                  1653, 317, 743, 918, 1236, 2179, 2808, 1472, 604, 2038, 738, 1918, 360, 2729, 108, 2286, 2427, 315,
                  266, 425, 2105, 1922, 480, 2851, 258, 1899, 2922, 2241, 1751, 1182, 1781, 2208, 1590, 1085, 2686,
                  1425, 2085, 2170, 234, 2131, 2873, 1482, 2956, 1987, 1481, 1691, 2361, 578, 1097, 2035, 2014, 125,
                  984, 2193, 350, 2362, 39, 269, 2335, 2079, 1409, 577, 2505, 1594, 1478, 1718, 1797, 1762, 1109, 2974,
                  1243, 2603, 249, 2223, 1735, 837, 1602, 1366, 1753, 866, 371, 1865, 2072, 1421, 899, 1692, 800, 404,
                  2854, 1664, 2150, 1782, 1068, 2910, 1601, 49, 666, 1928, 434, 495, 1741, 1018, 854, 352, 2285, 24,
                  2566, 2483, 1597, 1851, 3034, 880, 2992, 1193, 1196, 1078, 533, 2732, 1412, 642, 647, 1315, 1058,
                  2586, 1927, 1775, 1392, 966, 413, 406, 1737, 373, 664, 1136, 1848, 1152, 1728, 79, 2385, 1929, 1895,
                  799, 212, 2203, 1167, 1891, 2618, 237, 2311, 599, 2103, 2013, 51, 325, 1125, 675, 40, 1833, 1766, 437,
                  2017, 1276, 1881, 1776, 9, 1631, 439, 2423, 2619, 1878, 826, 2186, 909, 2943, 670, 1110, 257, 649,
                  2213, 1712, 1497, 1088, 2761, 517, 136, 20, 2316, 2730, 754, 1411, 1170, 860, 1326, 1686, 70, 1641,
                  2056, 54, 672, 2166, 1897, 2065, 1618, 2591, 1620, 1405, 33, 2439, 2383, 2572, 2458, 132, 697, 896,
                  576, 2954, 19, 911, 370, 433, 544, 2762, 4, 2433, 920, 2457, 1855, 1930, 1739, 1443, 501, 383, 1137,
                  268, 2418, 2198, 2211, 1901, 1264, 2175, 2925, 60, 2273, 1879, 2212, 2674, 874, 1062, 2256, 2482,
                  2856, 2967, 1277, 1510, 429, 2104, 299, 1828, 1072, 871, 552, 1166, 1012, 247, 851, 474, 2519, 869,
                  2414, 2460, 2136, 1381, 1234, 1327, 2366, 1371, 801, 3025, 1244, 2939, 2347, 687, 144, 1887, 2878,
                  196, 2718, 1904, 1479, 1588, 1534, 1254, 958, 1871, 1540, 1426, 2064, 2509, 755, 767, 2827, 1711, 68,
                  301, 2516, 398, 135, 852, 3022, 1038, 1880, 1202, 1475, 2147, 2154, 947, 2265, 917, 944, 2928, 1868,
                  856, 1662, 1854, 438, 1162, 218, 120, 778, 2096, 367, 1622, 436, 17, 2589, 324, 1309, 550, 1107, 1862,
                  106, 1906, 1724, 2512, 2398, 2907, 86, 337, 785, 1538, 689, 146, 729, 2644, 2336, 1980, 529, 1933,
                  1511, 1655, 1872, 1869, 1629, 246, 1553, 980, 1533, 1938, 2246, 2665, 1089, 423, 2927, 2044, 1952,
                  417, 472, 1260, 1941, 1902, 1508, 2964, 2161, 358, 1574, 531, 493, 2120, 92, 1757, 3108, 1019, 1984,
                  1648, 462, 1911, 1956, 1843, 1554, 698, 424, 69, 84, 2496, 873, 2197, 1297, 465, 319, 2868, 2692, 545,
                  1091, 446, 347, 981, 1400, 2543, 1774, 1269, 351, 2576, 722, 1671, 949, 1336, 564, 2556, 1592, 1589,
                  1361, 2331, 2291, 2879, 814, 281, 2142, 526, 1049, 1565, 2711, 2342, 1796, 48, 1296, 793, 1638, 1406,
                  2898, 1258, 274, 2108, 2205, 2646, 721, 2480, 825, 865, 2532, 1321, 2476, 2, 403, 415, 1486, 2215,
                  1146, 2250, 168, 1809, 302, 127, 598, 2497, 456, 2171, 2216, 1804, 2119, 1659, 1579, 1390, 972, 2055,
                  2158, 1369, 400, 948, 507, 828, 1473, 3005, 615, 29, 2084, 2681, 1024, 1057, 995, 1402, 1121, 139,
                  1689, 663, 1016, 513, 1134, 1803, 798, 318, 1074, 1468, 1561, 1453, 2833, 1656, 2021, 486, 1161, 1461,
                  1969, 1261, 1349, 1889, 1644, 2757, 1362, 2859, 1909, 898, 137, 1598, 2538, 1141, 1572, 1480, 1603,
                  2352, 1853, 2419, 1988, 1949, 2525, 509, 945, 1149, 344, 1668, 372, 1874, 631, 956, 163, 887, 2863,
                  1654, 2225, 1266, 760, 362, 2503, 1344, 202, 537, 1792, 2829, 2831, 1509, 2590, 1630, 1206, 1683, 759,
                  2977, 300, 1944, 95, 2911, 1636, 1367, 1231, 1467, 270, 2528, 2042, 524, 2219, 2779, 1151, 1399, 162,
                  2472, 1673, 1850, 226, 1181, 2461, 1877, 1325, 1916, 1010, 1083, 662, 3024, 10, 2181, 2675, 1289, 720,
                  2164, 2664, 1663, 2249, 971, 1945, 2607, 2087, 199, 426, 768, 1740, 725, 1235, 2057, 1331, 117, 1318,
                  1493, 739, 1729, 2749, 1523, 703, 590, 1071, 1428, 528, 2596, 1575, 306, 1621, 586, 2529, 43, 2660,
                  1640, 1760, 2303, 1395, 2971, 473, 635, 164, 1469, 771, 1413, 992, 1587, 1939, 265, 2299, 1014, 1608,
                  2149, 2670, 562, 746, 2595, 1532, 64, 1515, 1031, 220, 674, 2410, 1284, 1291, 557, 339, 932, 2717,
                  1045, 669, 1530, 1450, 2743, 151, 453, 1699, 525, 466, 201, 1821, 1446, 1912, 1965, 2333, 1154, 2805,
                  1993, 2464, 1609, 2355, 1950, 1307, 1011, 1970, 1761, 761, 1228, 2565, 2478, 375, 1675, 2580, 432,
                  2485, 1742, 240, 1749, 2194, 357, 3109, 1100, 443, 463, 633, 1313, 2495, 2792, 110, 1658, 538, 827,
                  216, 167, 1102, 3008, 813, 2950, 1564, 1240, 2001, 405, 2569, 1745, 1903, 1448, 2582, 1251, 2122,
                  2341, 2406, 2570, 915, 2023, 2558, 2247, 1377, 2473, 407, 1358, 2243, 692, 634, 408, 2604, 2277, 1617,
                  342, 2068, 1764, 858, 701, 1124, 1756, 2140, 505, 1294, 272, 47, 1437, 806, 242, 2940, 1619, 1391,
                  1265, 1432, 570, 1759, 1385, 1052, 1115, 2858, 2477, 1549, 2157, 1661, 979, 1139, 938, 154, 2009, 522,
                  1247, 37, 1805, 1140, 1571, 2308, 343, 1907, 1754, 1008, 1050, 1122, 1585, 1343, 730, 885, 558, 750,
                  1768, 859, 1165, 296, 90, 2363, 1246, 172, 42, 2238, 2876, 805, 987, 1477, 601, 1680, 1924, 2545,
                  2608, 1063, 213, 1059, 2614, 2499, 2828, 1186, 3019, 457, 1275, 284, 1111, 2135, 752, 2918, 1263,
                  2130, 790, 2280, 233, 2330, 2500, 1795, 2082, 834, 882, 2069, 811, 1866, 857, 2300, 1807, 2568, 3007,
                  1370, 1459, 788, 982, 640, 1544, 1082, 385, 2364, 929, 864, 1389, 2420, 288, 210, 1127, 605, 822, 849,
                  2573, 862, 593, 1073, 1087, 2752, 2870, 733, 2599, 2006, 1147, 229, 1387, 2102, 1635, 782, 646, 855,
                  3014, 469, 2864, 742, 277, 1189, 2076, 1667, 483, 174, 797, 661, 2700, 521, 1079, 2400, 2487, 506,
                  523, 2012, 389, 965, 1305, 1723, 66, 1084, 1591, 1915, 629, 1023, 1788, 1270, 2195, 203, 1403, 565,
                  2540, 1430, 2978, 821, 817, 3011, 365, 667, 1457, 935, 2932, 710, 3106, 1242, 1352, 1002, 802, 659,
                  964, 2536, 2375, 1224, 1623, 1485, 2437, 97, 2318, 191, 1070, 996, 1914, 2008, 1847, 555, 656, 1487,
                  1198, 1780, 412, 311, 397, 1322, 1288, 1548, 2656, 93, 56, 1303, 133, 1650, 243, 928, 2340, 2124, 419,
                  2493, 259, 1163, 8, 391, 2979, 374, 1831, 2258, 833, 2201, 1094, 169, 1431, 2492, 2074, 1359, 970,
                  883, 63, 88, 322, 1195, 2236, 1133, 2146, 853, 2578, 67, 65, 2148, 348, 1992, 2753, 2822, 489, 2588,
                  1219, 719, 1639, 390, 2837, 1627, 369, 1342, 1886, 2521, 1883, 1963, 1061, 116, 2583, 1798, 2794, 677,
                  1665, 1001, 1546, 902, 2631, 1188, 844, 2834, 1143, 395, 327, 148, 1169, 772, 2110, 2018, 1713, 2328,
                  392, 173, 1772, 2245, 2567, 1528, 1496, 1210, 1440, 2877, 1632, 477, 2924, 1255, 2754, 789, 83, 58,
                  126, 1398, 580, 786, 923, 2287, 2380, 1900, 630, 2357, 819, 25, 2327, 943, 1531, 2059, 36, 2052, 2706,
                  1611, 184, 387, 1209, 215, 2812, 1438, 2407, 1606, 748, 1221, 2434, 1697, 1157, 1458, 2229, 658, 189,
                  1207, 1717, 571, 1666, 2244, 832, 2869, 402, 3002, 115, 2462, 338, 1280, 1464, 57, 1674, 2955, 276,
                  130, 1917, 30, 222, 1777, 792, 482, 158, 715, 1213, 208, 1153, 2658, 1975, 1042, 1004, 2369, 2304,
                  1483, 333, 2650, 1578, 1973, 1834, 624, 520, 1095, 1520, 1894, 2139, 1104, 1075, 155, 2391, 652, 1576,
                  177, 2162, 411, 1217, 27, 1936, 241, 636, 1130, 359, 1502, 1274, 758, 1259, 1961, 1607, 1211, 1722,
                  632, 2506, 175, 2237, 639, 1948, 271, 1682, 2564, 2015, 2731, 1512, 3033, 2267, 2563, 2719, 546, 104,
                  942, 305, 2745, 1310, 354, 2003, 1180, 1060, 2450, 796, 2575, 1129, 1227, 1273, 363, 1176, 884, 1738,
                  1000, 1216, 1859, 78, 364, 2294, 1379, 89, 2768, 696, 192, 691, 2951, 245, 1758, 171, 2314, 1048,
                  1380, 1003, 1839, 2785, 320, 179, 2264, 2187, 2930, 2494, 102, 2054, 597, 2984, 2431, 1415, 1985, 835,
                  458, 731, 2413, 2849, 295, 3015, 2206, 452, 200, 1036, 2725, 1043, 660, 2896, 2874, 2836, 914, 85,
                  1015, 1350, 962, 2309, 142, 244, 1436, 2826, 1708, 617, 304, 1535, 209, 891, 1384, 803, 1009, 2838,
                  1017, 1527, 1563, 961, 1065, 1316, 1456, 1725, 901, 573, 2235, 678, 2511, 1447, 2107, 2379, 519, 382,
                  764, 1702, 484, 396, 1888, 435, 2969, 2623, 2442, 2481, 1233, 1669, 1272, 936, 2486, 614, 211, 1836,
                  2202, 1649, 2742, 1022, 2571, 2070, 582, 111, 3006, 2609, 2533, 1817, 1529, 1051, 98, 2659, 2382, 323,
                  2184, 1505, 2371, 157, 2447, 1330, 2141, 1040, 1208, 2913, 1462, 1784, 1787, 619, 900, 1225, 1873,
                  1946, 1155, 643, 530, 1651, 1148, 1827, 2221, 330, 2315, 2635, 53, 2819, 206, 1404, 2178, 379, 2759,
                  2125, 1474, 2466, 2188, 1215, 427, 440, 673, 1543, 968, 2210, 1685, 1376, 428, 781, 298, 1239, 2071,
                  2973, 273, 1892, 335, 278, 2705, 2217, 2471, 3023, 941, 2050, 76, 2129, 2324, 2465, 579, 355, 592,
                  2802, 1856, 1811, 1439, 1688, 1444, 420, 2577, 2114, 1281, 2368, 870, 1614, 1763, 934, 2553, 494,
                  1861, 950, 1721, 1964, 1132, 176, 1625, 2628, 1150, 940, 726, 1791, 794, 1526, 96, 1290, 264, 1423,
                  1230, 991, 1521, 963, 711, 81, 957, 1222, 2255, 1913, 2455, 224, 1596, 2554, 2518, 683, 61, 2222, 959,
                  44, 468, 1996, 23, 1838, 421, 2116, 2259, 903, 1846, 1858, 2113, 655, 2630, 1106, 1646, 2991, 845,
                  1744, 1311, 714, 1844, 1020, 2242, 393, 1420, 1953, 1979, 848, 1184, 321, 1314, 569, 539, 32, 861,
                  1131, 839, 1410, 59, 1278, 763, 1175, 1876, 600, 2548, 1678, 774, 2894, 1306, 1519, 1704, 2153, 1340,
                  808, 910, 2942, 3, 349, 2479, 2067, 2356, 145, 314, 1237, 2490, 460, 618, 2338, 227, 1695, 1849, 2372,
                  575, 1994, 471, 2550, 1752, 2305, 122, 1730, 1328, 713, 2046, 1375, 2594, 908, 622, 1252, 185, 1882,
                  161, 100, 1120, 732, 511, 1921, 1726, 409, 214, 2893, 461, 1114, 326, 2094, 2172, 2999, 2312, 990,
                  2239, 897, 594, 1433, 71, 810, 2522, 451, 2585, 951, 464, 2401, 875, 784, 1701, 2432, 1285, 332, 2348,
                  1113, 1615, 1932, 444, 2376, 2106, 1860, 2389, 114, 1108, 107, 2292, 1190, 2716, 1119, 626, 2033,
                  1429, 476, 1076, 1567, 2592, 1218, 727, 2508, 876, 1054, 2748, 831, 919, 2226, 2645, 2549, 254, 2688,
                  516, 418, 1545, 1332, 1200, 1442, 2098, 2671, 747, 1360, 2689, 1382, 1183, 448, 892, 2251, 2903, 2698,
                  1810, 159, 262, 75, 99, 1605, 836, 2865, 399, 2086, 3009, 913, 997, 2620, 2041, 255, 907, 737, 1348,
                  648, 702, 1628, 2284, 2073, 878, 1977, 170, 657, 953, 180, 872, 2080, 280, 2653, 2278, 225, 2990,
                  1156, 2395, 1256, 1813, 1027, 2800, 1086, 1707, 2726, 559, 2089, 1451, 2855, 239, 2156, 384, 1997,
                  905, 1823, 1910, 217, 2474, 912, 294, 1926, 1028, 881, 1053, 1719, 595, 1013, 2412, 1471, 2323, 450,
                  2695, 2682, 1245, 572, 1826, 353, 2621, 1021, 585, 1105, 166, 1047, 1710, 1494, 2444, 1329, 2915, 500,
                  2587, 293, 706, 12, 2498, 15, 512, 1454, 2895, 2253, 863, 316, 1117, 2843, 250, 2713, 129, 960, 2025,
                  2517, 2899, 2900, 2988, 2557, 307, 2100, 2359, 2639, 2962, 105, 508, 2155, 1424, 221, 1793, 2275,
                  1145, 688, 252, 2426, 2504, 1434, 2989, 2112, 563, 1504, 2043, 1676, 2610, 488, 228, 690, 2093, 2763,
                  1559, 815, 2334, 2680, 1513, 1449, 2263, 1568, 223, 757, 777, 1338, 1990, 1660, 1957, 1732, 540, 504,
                  2906, 2167, 2097, 2190, 1250, 791, 1842, 487, 628, 1160, 2654, 310, 2775, 922, 2562, 2039, 745, 549,
                  2484, 668, 1317, 1268, 28, 1253, 2912, 1346, 955, 1286, 2321, 1135, 2755, 773, 1581, 1355, 492, 904,
                  2456, 2279, 2144, 366, 31, 2396, 2099, 2996, 1335, 46, 1, 1503, 1972, 1241, 2872, 3018, 1864, 2415,
                  2090, 138, 455, 812, 2134, 1800, 2643, 770, 1393, 2735, 561, 1158, 2624, 2953, 2351, 568, 380, 1029,
                  2673, 2446, 74, 230, 2783, 1103, 2741, 2405, 608, 1518, 924, 2923, 410, 765, 1419, 2625, 286, 930,
                  1164, 1593, 1799, 2736, 55, 1096, 620, 681, 1516, 11, 740, 685, 1347, 2641, 1187, 2350, 1765, 3000,
                  2534, 734, 497, 101, 2424, 644, 194, 1364, 2995, 1026, 204, 1986, 2289, 780, 2397, 103, 2295, 251,
                  1308, 830, 3030, 2491, 2891, 2329, 641, 1025, 16, 2394, 976, 1786, 2809, 708, 140, 952, 2436, 1971,
                  1501, 2881, 2293, 2261, 2428, 775, 1118, 2183, 2648, 394, 842, 518, 2821, 1736, 542, 584, 197, 682,
                  2358, 1700, 3021, 94, 2793, 1542, 2449, 181, 2734, 2337, 312, 1351, 820, 2469, 1174, 279, 766, 1041,
                  1098, 916, 291, 2959, 124, 2075, 1378, 602, 1407, 2694, 2530, 510, 1556, 2633, 993, 973, 34, 3102,
                  1633, 2270, 612, 1705, 1324, 651, 178, 1126, 2687, 1032, 1991, 160, 45, 7, 2176, 2728, 1852, 2199,
                  1734, 1374, 1524, 2117, 2016, 336, 2817, 2502, 2230, 490, 2723, 2778, 134, 156, 2862, 933, 2489, 596,
                  1498, 974, 1801, 1829, 1394, 2707, 2926, 2200, 2218, 988, 35, 939, 2579, 1998, 716, 285, 283, 1919,
                  1962, 1194, 502, 2853, 2061, 2696, 2454, 1492, 2945, 975, 2463, 1299, 1301, 1814, 1943, 1989, 2373,
                  1999, 1830, 1893, 829, 503, 2332, 879, 1372, 187, 1837, 341, 2030, 3004, 1905, 2672, 2233, 847, 2551,
                  1560, 3104, 2825, 978, 1974, 1466, 267, 671, 2048, 2593, 551, 954, 1967, 3103, 925, 2290, 128, 1408,
                  2875, 2637, 2262, 1037, 1816, 2468, 989, 621, 1229, 894, 2724, 2710, 977, 2699, 787, 235, 2019, 889,
                  2634, 532, 611, 2475, 2850, 983, 2005, 684, 2209, 1634, 119, 736, 289, 1345, 1819, 2693, 198, 609, 38,
                  2941, 2835, 581, 195, 1767, 705, 1101, 867, 2737, 2946, 823, 2381, 560, 2010, 2320, 999, 1007, 2257,
                  2993, 2307, 2392, 2640, 2663, 665, 2453, 2886, 2602, 2388, 723, 2339, 2344, 1465, 1090, 2617, 1577,
                  2228, 1867, 2126, 2326, 377, 2204, 2111, 700, 2750, 1414, 2378, 238, 445, 1733, 1822, 2408, 1931,
                  2811, 1386, 22, 1293, 2600, 553, 2961, 2840, 2531, 741, 2555, 2709, 3105, 1875, 2416, 638, 2574, 890,
                  441, 3032, 1960, 2276, 2266, 587, 1488, 637, 2544, 937, 1569, 1034, 2151, 1363, 1238, 807, 2024, 893,
                  2180, 188, 1785, 3101, 2173, 2260, 2384, 2207, 2814, 2703, 13, 1099, 2948, 654, 2448, 1092, 1214,
                  2045, 1968, 613, 2679, 356, 2701, 491, 2367, 2231, 606, 1005, 459, 2697, 2234, 1006, 2020, 1955, 2839,
                  2268, 2354, 2902, 309, 470, 541, 2404, 676, 2007, 2842, 2310, 2766, 2629, 1825, 2901, 2248, 2527, 868,
                  2281, 1418, 2963, 926, 843, 2152, 80, 2185, 2714, 2288, 2520, 1958, 2036, 2060, 1832, 2892, 1066,
                  2559, 2537, 2298, 2668, 2047, 2390, 1750, 1978, 2657, 414, 2165, 2615, 2174, 193, 2425, 2220, 2137,
                  1982, 3026, 2758, 2501, 1983, 2402, 2417, 2947, 840, 3107]

vehicle_i80 = [2224, 2283, 2252, 2388, 1919, 2032, 3027, 2089, 2132, 2109, 1869, 3214, 2329, 2374, 3009, 2181, 2051,
               1871, 1807, 2460, 3199, 2321, 2330, 2136, 1946, 2077, 1809, 3159, 2215, 3314, 1849, 2497, 1925, 2184,
               2738, 2165, 1775, 2105, 2149, 2590, 3128, 2722, 3031, 2265, 2609, 1924, 2994, 2435, 1910, 2991, 1895,
               2256, 2633, 2595, 3010, 2474, 1887, 2036, 2400, 2099, 2486, 2333, 2809, 3221, 2273, 2846, 1889, 1951,
               2799, 2209, 2542, 2639, 2328, 2241, 2303, 3021, 2337, 2984, 2319, 2752, 2641, 2941, 2020, 2296, 1878,
               3233, 2519, 2491, 3155, 3309, 1863, 1796, 2475, 2059, 2076, 1900, 2500, 2368, 2234, 3273, 3002, 2153,
               1767, 1943, 2016, 1825, 2431, 2463, 3333, 3063, 2054, 3321, 2186, 2254, 1941, 1774, 2362, 2162, 2277,
               2190, 2133, 2119, 2532, 2262, 3237, 1974, 2116, 2297, 2425, 2527, 3006, 2520, 2158, 2069, 2156, 2725,
               3249, 2260, 2700, 2015, 2512, 2468, 2199, 2416, 1969, 2312, 1885, 2011, 3257, 2314, 2264, 2348, 2205,
               2989, 2702, 3156, 2245, 2029, 1881, 2307, 2074, 2438, 2189, 2211, 3181, 2332, 2168, 2531, 2293, 1834,
               2071, 1967, 2423, 2934, 2083, 1799, 2138, 2480, 2082, 3326, 2259, 1827, 1935, 2212, 2454, 2434, 2030,
               2143, 2216, 3061, 3320, 2464, 3352, 2413, 1957, 1854, 3209, 2213, 2560, 2448, 2823, 2917, 2447, 2498,
               2301, 2985, 2087, 2488, 3272, 2240, 1858, 3046, 2953, 1908, 1938, 3078, 2208, 1907, 1873, 2048, 2325,
               2141, 2355, 2547, 2146, 3300, 2918, 2349, 2360, 2998, 2387, 1915, 2997, 2103, 2674, 2528, 2727, 2351,
               3109, 2614, 2372, 1853, 2402, 3232, 2123, 2343, 2511, 2433, 2796, 2300, 2131, 2172, 2398, 1763, 2370,
               2453, 2236, 2173, 3355, 3226, 2145, 2816, 3013, 2278, 1772, 2383, 2837, 2657, 3040, 2516, 2755, 2115,
               2401, 2379, 2164, 2399, 1828, 2005, 2947, 2203, 2320, 2098, 2861, 3179, 2375, 2841, 1851, 2366, 2118,
               2688, 2056, 2253, 3305, 2148, 2807, 2112, 2340, 2152, 2137, 3304, 2392, 1960, 2449, 2151, 2107, 2681,
               2282, 2513, 2160, 2751, 2285, 2113, 2922, 1933, 3008, 2915, 2376, 3022, 3168, 2266, 1806, 2044, 3299,
               2121, 3222, 3361, 2309, 3044, 2536, 2515, 2457, 1793, 3004, 2404, 1792, 2646, 2412, 2012, 2188, 2095,
               2698, 1810, 2458, 2286, 3073, 2421, 2611, 2473, 2567, 2233, 2863, 2142, 1788, 3261, 3038, 2944, 2427,
               2385, 3277, 2061, 2304, 3111, 3307, 2686, 2281, 2518, 2018, 2414, 2007, 2601, 1940, 2422, 1765, 2521,
               2072, 2446, 2395, 3254, 2364, 1914, 3033, 2938, 2932, 2811, 1801, 2243, 1886, 2561, 2930, 3125, 2780,
               1824, 2766, 3051, 2381, 1831, 2411, 3284, 3001, 2541, 3223, 2615, 1813, 2483, 2218, 2544, 2455, 1921,
               2175, 2090, 2130, 2605, 3085, 2306, 2479, 1884, 2913, 3122, 2200, 2369, 2022, 2356, 2770, 2525, 2443,
               3208, 2193, 2680, 3000, 1995, 3136, 2220, 2140, 2503, 2122, 3115, 2558, 2948, 3279, 2263, 3057, 2170,
               2783, 2484, 3288, 2514, 2462, 3084, 2363, 2845, 3103, 2872, 1769, 1835, 3020, 2102, 2850, 1762, 2251,
               2185, 3129, 2002, 2334, 2612, 2261, 2642, 2410, 3041, 1994, 1923, 2635, 2819, 2602, 3018, 2509, 1909,
               1764, 2415, 2824, 2589, 2973, 2052, 2703, 3101, 2670, 3005, 3134, 2459, 1879, 2246, 3317, 3324, 1818,
               3123, 2552, 1821, 2258, 2868, 3212, 3308, 2040, 2865, 2326, 3059, 2227, 1861, 2078, 2417, 2487, 1899,
               2024, 2923, 2389, 2699, 2124, 1848, 1844, 2929, 2023, 1918, 2197, 2794, 2194, 2758, 2439, 1891, 2786,
               2195, 3090, 2100, 3024, 3342, 1874, 2445, 2391, 2049, 1953, 2972, 2782, 2110, 2008, 3080, 2101, 2358,
               2742, 2585, 3120, 3158, 1962, 2478, 3169, 3121, 2825, 2247, 2467, 2935, 2718, 3088, 3014, 2144, 2064,
               2456, 3322, 2280, 2384, 3126, 3167, 3265, 3139, 2728, 3227, 1758, 2668, 3323, 2485, 2017, 2634, 2974,
               2004, 2539, 2822, 2848, 1901, 2232, 2311, 1897, 2640, 1961, 3354, 2274, 2858, 2584, 2318, 3100, 2916,
               3148, 3242, 2342, 2371, 2436, 2538, 1932, 2791, 2476, 2365, 1893, 2426, 2694, 2088, 2134, 1860, 1894,
               2027, 2039, 1952, 3287, 3291, 2065, 2060, 1999, 2031, 2924, 2675, 3294, 1968, 3093, 2169, 1819, 2444,
               2111, 3007, 1800, 1956, 2537, 2928, 2499, 3219, 1773, 2267, 3353, 2851, 2693, 1795, 1942, 2176, 1876,
               2219, 2221, 2063, 1836, 2287, 2159, 3118, 2969, 2288, 2982, 2292, 2026, 2108, 2289, 2931, 2744, 2914,
               3357, 3263, 1958, 3172, 3290, 2812, 3035, 3274, 2999, 2481, 2701, 1840, 2394, 2305, 2946, 2550, 1771,
               2275, 2717, 3349, 3286, 2661, 2198, 2835, 2720, 1841, 2730, 2775, 3363, 1977, 1862, 1996, 2308, 2155,
               2272, 1963, 2724, 2469, 2352, 2690, 2396, 2553, 2183, 1777, 2772, 3153, 2952, 3313, 2652, 2248, 3362,
               2842, 2731, 2714, 2743, 1975, 2085, 2697, 2534, 1857, 3094, 1906, 2429, 2135, 3065, 1803, 2210, 2057,
               2424, 2853, 2572, 3296, 3239, 2576, 3154, 3247, 2441, 1847, 3289, 2546, 2161, 2510, 3003, 2354, 3260,
               2393, 2829, 2187, 2322, 3097, 3138, 1859, 2980, 2139, 2787, 1787, 3124, 1850, 2616, 2117, 2654, 1781,
               2242, 3050, 3055, 3117, 1761, 1814, 2662, 2094, 3141, 2767, 3238, 3194, 3114, 2860, 1802, 2276, 2667,
               3108, 2174, 2793, 1998, 2298, 2192, 2776, 3086, 2594, 2927, 2808, 3019, 1888, 2638, 3015, 2704, 1832,
               2403, 2338, 3198, 3032, 2106, 2390, 2976, 2377, 1839, 3210, 2788, 2769, 2506, 1797, 1904, 2021, 2711,
               2014, 2583, 1833, 2420, 2080, 2302, 1890, 3193, 3281, 1868, 1808, 3217, 3144, 2405, 3049, 3025, 3191,
               1966, 2093, 2239, 2543, 2522, 1766, 2726, 2335, 3188, 2290, 2548, 1905, 2081, 3211, 2828, 2466, 2945,
               2470, 2582, 1997, 2739, 2428, 2050, 2344, 2568, 2630, 2817, 2357, 2230, 2206, 3174, 2962, 2733, 3343,
               3095, 2128, 2407, 1973, 1959, 3253, 2958, 1817, 2656, 2563, 1911, 2238, 3248, 3053, 2986, 2442, 3325,
               2975, 2818, 1930, 1931, 3034, 2968, 2180, 1934, 2933, 2517, 3215, 2867, 1912, 1972, 2452, 3196, 2033,
               3102, 3245, 2432, 2339, 2862, 1856, 3252, 2202, 1926, 2695, 2430, 2618, 2440, 1768, 1917, 2940, 2237,
               3280, 1798, 3133, 2217, 1920, 2062, 2555, 2949, 3240, 3150, 1757, 2591, 2084, 2501, 2956, 2797, 3366,
               2669, 2830, 2996, 3340, 3315, 2086, 2359, 3175, 2437, 3301, 2963, 2774, 3295, 2327, 2987, 3230, 2939,
               2857, 2341, 2013, 2070, 1805, 3060, 2347, 2235, 3054, 3077, 2773, 2566, 1976, 1877, 2270, 2163, 2678,
               3285, 3225, 1922, 1964, 3036, 2171, 3152, 3067, 2147, 2034, 2978, 3011, 2294, 2554, 2079, 3180, 2075,
               3163, 2166, 2207, 2764, 3319, 2827, 3262, 3258, 3275, 2250, 2943, 3183, 2222, 2838, 3297, 2346, 2624,
               2201, 3072, 2257, 1913, 2324, 3074, 3016, 2813, 1993, 2055, 3066, 2001, 2120, 3143, 3187, 2942, 3264,
               3229, 3318, 2035, 2053, 2992, 3091, 3200, 3089, 3113, 3135, 2745, 3255, 3312, 3092, 2150, 3076, 2406,
               2608, 3302, 3360, 3176, 1780, 2993, 2689, 3306, 3345, 2831, 3246, 2708, 3026, 2955, 2683, 2864, 2871,
               3186, 3110, 2866, 2378, 1867, 2912, 3364, 1882, 3206, 3270, 2477, 3216, 2502, 1830, 2508, 3197, 2869,
               1838, 1812, 3244, 2597, 2599, 3140, 1852, 2493, 3178, 3228, 2045, 2073, 2041, 3256, 2575, 1804, 2268,
               3064, 2636, 2847, 3170, 2091, 2586, 3127, 3039, 2574, 3278, 3166, 3071, 2957, 2361, 2856, 3146, 2066,
               3282, 2622, 3271, 2556, 2687, 3162, 3365, 3327, 1896, 1939, 1945, 3171, 2685, 1937, 2003, 3293, 1823,
               3205, 2127, 3344, 2037, 3151, 1903, 2315, 2472, 2870, 3173, 2990, 3259, 2937, 1971, 3292, 3132, 2911,
               2097, 1811, 1955, 868, 2490, 1485, 1479, 979, 1440, 1349, 925, 1565, 1541, 880, 764, 583, 251, 1302,
               1717, 1551, 959, 349, 541, 1700, 1246, 372, 219, 1480, 1203, 2196, 839, 1703, 1215, 1731, 1418, 5, 1362,
               438, 1519, 1307, 772, 1022, 556, 1261, 1043, 1125, 423, 1407, 507, 1406, 725, 587, 430, 394, 815, 830,
               1392, 1613, 1673, 326, 1659, 1569, 775, 1698, 1126, 759, 1084, 456, 647, 373, 677, 286, 1749, 977, 937,
               1681, 1319, 666, 1626, 429, 1383, 222, 1595, 108, 9, 2910, 389, 723, 1439, 297, 1423, 944, 461, 1217,
               1668, 1514, 1706, 1317, 889, 1396, 1053, 1324, 1639, 241, 521, 973, 614, 1612, 501, 1436, 2889, 1898,
               1648, 574, 216, 995, 1725, 1121, 459, 315, 726, 1662, 1321, 1112, 1359, 1106, 641, 439, 846, 1518, 1265,
               1597, 714, 689, 1573, 397, 730, 950, 859, 185, 1308, 306, 1550, 328, 61, 1153, 2067, 1736, 391, 1730,
               1405, 561, 1023, 1162, 412, 1042, 1281, 1596, 1390, 1199, 1842, 1570, 378, 1630, 314, 54, 1633, 1133,
               779, 308, 141, 490, 989, 234, 1610, 1386, 1373, 1751, 1458, 746, 1691, 1379, 1361, 1164, 549, 1341, 1631,
               1603, 1283, 932, 449, 924, 1490, 416, 1497, 1864, 809, 1225, 782, 678, 1656, 619, 1347, 1292, 1467, 762,
               1011, 856, 1365, 1522, 227, 1343, 499, 910, 444, 1093, 1568, 828, 40, 1709, 536, 72, 740, 106, 860, 538,
               2855, 1503, 1330, 59, 1543, 1712, 980, 1657, 325, 515, 155, 1987, 258, 1521, 41, 784, 248, 510, 1026,
               1072, 1205, 1378, 752, 1380, 213, 356, 1120, 1722, 905, 706, 607, 1759, 1664, 366, 793, 1200, 2881, 1381,
               36, 1073, 943, 1388, 1614, 941, 1413, 282, 204, 1558, 175, 1305, 269, 1299, 1462, 421, 4, 1666, 2409,
               2771, 2885, 2104, 237, 1950, 1255, 1500, 253, 1555, 1027, 1623, 497, 2874, 1031, 522, 1327, 1794, 358,
               870, 1576, 2836, 898, 525, 1719, 1366, 476, 492, 671, 1415, 1036, 1715, 1279, 1320, 1316, 1403, 1410,
               2380, 212, 334, 1163, 1756, 147, 1306, 440, 442, 1645, 601, 1276, 1578, 42, 1257, 711, 1674, 1463, 938,
               824, 1222, 820, 1744, 174, 866, 2114, 1019, 996, 1287, 1017, 96, 790, 1586, 1105, 1437, 623, 897, 1535,
               1326, 483, 6, 1489, 1642, 1584, 1178, 1511, 727, 1714, 1048, 136, 757, 32, 79, 651, 1367, 1755, 377, 454,
               1829, 1985, 640, 1227, 1397, 571, 86, 299, 951, 2803, 343, 1049, 2010, 1428, 1745, 1421, 224, 876, 236,
               1262, 896, 400, 1704, 160, 173, 1101, 1494, 710, 346, 1297, 218, 283, 1618, 681, 612, 115, 2323, 52,
               1647, 530, 362, 861, 646, 1579, 970, 1232, 1151, 1979, 8, 508, 435, 1139, 1954, 1588, 1286, 1540, 1688,
               1424, 982, 844, 1574, 223, 1594, 637, 597, 1064, 1984, 318, 912, 974, 697, 822, 1450, 986, 958, 1502,
               434, 1224, 1455, 1460, 1431, 116, 1038, 1096, 506, 948, 639, 226, 44, 1229, 233, 895, 395, 385, 1314,
               1654, 2768, 934, 1536, 1173, 1488, 1370, 1087, 509, 256, 953, 1605, 687, 1456, 1770, 1650, 1592, 914,
               756, 877, 388, 1014, 534, 1661, 1143, 751, 1970, 753, 1682, 893, 338, 337, 1524, 1422, 1411, 683, 1002,
               477, 1822, 182, 487, 918, 2759, 1420, 43, 1005, 169, 1433, 981, 1193, 865, 1457, 1496, 1667, 1296, 1445,
               1363, 1676, 604, 1534, 102, 495, 776, 733, 191, 1399, 1572, 664, 920, 1471, 1122, 572, 10, 1739, 1069,
               1157, 1175, 1000, 1079, 1728, 1065, 1815, 2904, 1056, 692, 1372, 789, 1635, 1268, 545, 823, 613, 1687,
               1159, 1734, 1076, 291, 1290, 1328, 1104, 1375, 1643, 375, 1916, 1172, 277, 539, 699, 1738, 749, 70, 923,
               1374, 354, 903, 1190, 1419, 811, 3, 1533, 964, 398, 1131, 1552, 766, 2757, 819, 81, 100, 1260, 1018, 151,
               1559, 1447, 1046, 831, 298, 1694, 455, 2895, 272, 1567, 77, 84, 1037, 146, 2820, 862, 1168, 1177, 390,
               906, 1487, 1601, 30, 1620, 1385, 1248, 575, 230, 978, 1353, 1696, 801, 276, 1747, 1340, 1249, 1548, 1109,
               1526, 1981, 1009, 629, 1226, 707, 1270, 720, 1515, 481, 1339, 1481, 124, 1451, 235, 1191, 1345, 517, 966,
               857, 1701, 255, 1448, 1729, 50, 289, 1707, 926, 309, 166, 888, 342, 1058, 485, 194, 2806, 465, 316, 1577,
               2043, 1607, 292, 1272, 1561, 482, 1677, 450, 618, 2019, 1575, 605, 98, 713, 2800, 1006, 1539, 909, 290,
               952, 552, 244, 778, 1080, 85, 140, 1902, 2814, 1028, 722, 1228, 994, 245, 1653, 2893, 1382, 1377, 591,
               1135, 1368, 843, 353, 321, 1068, 1789, 29, 1598, 803, 900, 1621, 1726, 468, 1400, 1414, 1493, 632, 1498,
               1054, 1582, 479, 2408, 472, 563, 491, 2125, 615, 672, 1277, 1786, 1615, 547, 1274, 1557, 1604, 1150, 82,
               1708, 777, 1408, 589, 735, 1699, 596, 198, 384, 458, 968, 600, 1086, 1012, 1409, 143, 1376, 284, 294,
               190, 935, 621, 1563, 1231, 496, 2781, 176, 665, 420, 21, 594, 413, 1675, 267, 578, 1434, 1644, 781, 576,
               1444, 742, 2834, 608, 111, 232, 2883, 1148, 1678, 2042, 1655, 46, 259, 1733, 1417, 1234, 1663, 1115,
               1202, 834, 466, 399, 407, 1295, 1581, 1426, 1300, 350, 503, 933, 2875, 1429, 63, 1325, 1589, 1354, 441,
               599, 1478, 1184, 37, 1735, 1237, 987, 162, 1591, 818, 1782, 1024, 193, 946, 2873, 626, 2810, 1427, 333,
               448, 927, 2006, 595, 1198, 1154, 609, 1679, 1041, 668, 680, 1384, 307, 504, 379, 1486, 798, 1398, 1965,
               812, 1233, 718, 929, 894, 1473, 1243, 806, 991, 2899, 1029, 1556, 1516, 622, 947, 1108, 280, 15, 845, 88,
               161, 1449, 1099, 1003, 401, 1754, 51, 685, 1746, 763, 132, 1081, 323, 1214, 1323, 1538, 1346, 1208, 1566,
               95, 962, 1267, 1194, 976, 1158, 882, 1035, 1713, 1435, 1137, 1358, 243, 1512, 109, 1288, 1044, 93, 1461,
               1684, 1102, 628, 1393, 1197, 930, 1750, 1671, 335, 1816, 78, 1170, 1213, 1779, 901, 537, 1624, 1116,
               1039, 1742, 858, 303, 2815, 642, 1130, 1660, 1329, 1651, 150, 1988, 99, 1583, 1470, 1303, 1040, 673,
               1606, 351, 1161, 1309, 1627, 1160, 113, 1016, 396, 1617, 518, 1185, 1454, 341, 1332, 340, 984, 210, 1182,
               1491, 1132, 1685, 2777, 559, 264, 1743, 1469, 1254, 1695, 383, 1008, 638, 694, 890, 331, 1549, 1015, 886,
               393, 675, 1311, 7, 1711, 1360, 956, 754, 748, 11, 1209, 386, 484, 1395, 469, 211, 1055, 2336, 478, 1216,
               1097, 310, 2451, 2350, 863, 667, 916, 2397, 768, 179, 1532, 715, 1554, 381, 1282, 848, 1441, 121, 564,
               691, 1430, 67, 737, 1697, 1103, 704, 195, 189, 747, 548, 535, 611, 1218, 514, 761, 1253, 1149, 816, 963,
               1866, 473, 1230, 526, 48, 864, 12, 1702, 370, 738, 1692, 1259, 871, 2901, 1322, 68, 1083, 1705, 1720,
               584, 1989, 2028, 670, 238, 1892, 945, 1285, 1402, 494, 1123, 1294, 1529, 570, 1690, 1599, 703, 1740, 712,
               2345, 1212, 1710, 1600, 103, 432, 1342, 1004, 1071, 553, 1446, 1611, 183, 460, 1113, 1089, 770, 922, 74,
               26, 1527, 445, 2906, 1252, 2849, 686, 1090, 257, 13, 2894, 1477, 424, 447, 498, 807, 2888, 1495, 650,
               1223, 1689, 1505, 654, 660, 1724, 1513, 427, 1784, 972, 1236, 658, 1110, 1432, 954, 419, 90, 266, 1141,
               1085, 603, 838, 1100, 1304, 731, 700, 729, 1301, 829, 2331, 1235, 1507, 841, 1183, 1196, 336, 1284, 300,
               1394, 813, 1210, 1219, 329, 302, 246, 1504, 1520, 1067, 112, 17, 887, 1312, 773, 661, 705, 1727, 702,
               1634, 585, 907, 975, 1352, 2465, 1134, 1062, 633, 1438, 75, 1114, 560, 164, 550, 1665, 1074, 1737, 1220,
               371, 1401, 1371, 546, 320, 355, 414, 787, 69, 165, 1013, 588, 1117, 1927, 360, 1245, 2908, 137, 1669,
               649, 2880, 339, 1146, 156, 739, 875, 1244, 1608, 854, 679, 107, 1045, 364, 684, 2897, 2821, 1525, 1776,
               1718, 533, 453, 1264, 92, 1587, 1333, 1391, 2179, 582, 1875, 1239, 562, 1542, 852, 1057, 1986, 767, 305,
               682, 1258, 2902, 1136, 1453, 296, 1593, 1140, 446, 1723, 1240, 1147, 1181, 1357, 993, 997, 1155, 542,
               451, 110, 939, 1947, 417, 1369, 1560, 908, 22, 1683, 409, 505, 631, 376, 1032, 252, 403, 76, 1331, 1468,
               1179, 348, 804, 832, 2461, 387, 60, 1571, 261, 1590, 247, 656, 1176, 317, 1748, 1180, 755, 1499, 474,
               652, 480, 2178, 1482, 2877, 1298, 1546, 2191, 627, 663, 1351, 2785, 422, 1883, 817, 39, 1459, 1275, 904,
               57, 990, 821, 1672, 104, 1195, 149, 1517, 293, 825, 884, 780, 1063, 126, 1547, 1484, 529, 1641, 573, 586,
               1318, 1138, 1716, 1443, 911, 1632, 795, 382, 1658, 1047, 1992, 357, 833, 1501, 1092, 58, 1628, 620, 1826,
               324, 262, 1077, 1070, 2765, 2046, 214, 380, 168, 1174, 38, 1752, 869, 1991, 636, 878, 1387, 1241, 1088,
               797, 1128, 1562, 205, 783, 1165, 1686, 500, 744, 1082, 827, 49, 1098, 368, 902, 765, 669, 443, 406, 263,
               1127, 181, 1078, 1271, 643, 1472, 1356, 1211, 2317, 1095, 758, 1171, 142, 644, 1273, 2492, 1066, 785,
               249, 1929, 724, 2792, 1338, 327, 1291, 799, 1034, 760, 135, 1442, 792, 942, 278, 1001, 969, 794, 1315,
               1978, 1030, 1640, 1350, 486, 101, 520, 1990, 1355, 217, 788, 985, 2058, 695, 1094, 659, 693, 1025, 1278,
               554, 120, 892, 814, 2471, 367, 1466, 540, 304, 577, 1790, 565, 1949, 2419, 66, 657, 431, 425, 617, 1404,
               1256, 271, 965, 1741, 826, 708, 1263, 64, 837, 580, 361, 899, 1845, 2182, 55, 796, 199, 1509, 27, 1007,
               2898, 2126, 128, 1145, 581, 295, 24, 344, 418, 279, 688, 402, 1732, 177, 1250, 352, 1364, 1425, 1310,
               467, 192, 676, 791, 653, 1528, 2373, 363, 1602, 433, 1483, 1289, 457, 1247, 1619, 1221, 1348, 158, 203,
               881, 1060, 567, 133, 598, 1337, 2833, 1649, 2009, 1129, 1625, 851, 206, 1508, 25, 528, 913, 129, 1189,
               1293, 1118, 242, 655, 719, 800, 1334, 1510, 698, 1061, 961, 1476, 347, 774, 519, 23, 919, 1636, 187, 374,
               285, 1166, 734, 2353, 835, 1646, 1206, 786, 532, 1280, 1585, 2313, 544, 1152, 65, 369, 610, 138, 197,
               1020, 1091, 616, 260, 1389, 579, 1010, 1169, 732, 936, 1506, 872, 436, 1791, 1452, 1269, 1412, 1880,
               1344, 240, 743, 998, 1553, 1753, 915, 1523, 1416, 322, 1156, 2367, 1629, 405, 301, 630, 330, 1670, 1564,
               983, 1616, 634, 1465, 1075, 645, 1111, 270, 392, 188, 148, 123, 1201, 332, 1948, 1721, 721, 850, 2826,
               83, 1051, 558, 2890, 130, 635, 452, 1474, 1492, 209, 999, 87, 717, 808, 928, 741, 512, 874, 690, 428,
               312, 2496, 1936, 1033, 855, 80, 1637, 662, 842, 97, 940, 2025, 125, 2299, 1785, 709, 988, 47, 463, 769,
               1021, 1207, 1537, 464, 1335, 716, 2907, 853, 873, 502, 1778, 475, 960, 1609, 122, 555, 1107, 2316, 957,
               771, 287, 359, 836, 1313, 810, 311, 516, 53, 847, 949, 2886, 1680, 254, 2047, 745, 1119, 2844, 94, 462,
               488, 2382, 281, 91, 200, 1475, 2154, 319, 89, 1544, 411, 470, 955, 2269, 931, 163, 606, 1638, 493, 625,
               1622, 2271, 566, 696, 365, 1242, 265, 527, 410, 274, 415, 885, 1944, 569, 2760, 1652, 202, 967, 2854,
               1464, 1167, 180, 1050, 701, 159, 114, 268, 1545, 117, 590, 1192, 557, 186, 1266, 127, 2092, 1188, 2096,
               624, 28, 231, 1052, 879, 1580, 1531, 2761, 2000, 805, 883, 1251, 802, 1, 728, 45, 750, 2157, 313, 513,
               1872, 1059, 1336, 2804, 2167, 201, 1238, 426, 992, 437, 1530, 489, 31, 144, 2879, 1693, 2, 867, 1760,
               250, 917, 511, 849, 1142, 1865, 921, 2129, 891, 840, 19, 2495, 1124]