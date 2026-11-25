# Các màn chưa học: mạng 8 màn, bước 9.998.336 so với các mạng khác

Chấm bằng `mario_ppo.evaluate`, 20 lượt, seed 2026, lấy mẫu hành động. Màn mạng 8 màn đã học (1-1, 2-1, 3-1, 3-2, 4-1, 5-1, 5-2, 6-1) không có ở đây.
"% màn" = x chia cho x của cờ theo `runs/level-lengths/level-lengths.csv`; dấu – là màn không đo được cờ (kết thúc bằng ống hoặc mê cung).

## Bảng tổng: x trung bình (trong ngoặc là x xa nhất)

| Màn | x cờ | 8 màn | 2 màn | chuyên 1-1 | chuyên 4-1 | chuyên 2-3 |
|---|---|---|---|---|---|---|
| 1-2 | – | 427 (972) | 256 (652) | 177 (965) | chưa đo | 355 (892) |
| 1-3 | 2425 | 430 (439) | 157 (420) | 416 (445) | 403 (747) | 292 (315) |
| 1-4 | 2244 | 232 (236) | 494 (561) | 233 (234) | chưa đo | 232 (232) |
| 2-2 | – | 874 (1435) | 322 (514) | 1096 (1618) | 822 (2163) | 1084 (1522) |
| 2-3 | 3593 | 259 (1081) | 192 (650) | 843 (1762) | 1024 (1792) | *đã học:* 3492 (3593), 18 cờ |
| 2-4 | 2245 | 344 (348) | 389 (416) | 291 (344) | 337 (347) | 314 (344) |
| 3-3 | 2409 | 462 (472) | 305 (323) | 322 (454) | chưa đo | 324 (326) |
| 3-4 | 2244 | 358 (524) | 358 (434) | 324 (521) | 270 (274) | 505 (1435) |
| 4-2 | – | 244 (346) | 263 (466) | 281 (466) | 249 (350) | 223 (466) |
| 4-3 | 2345 | 398 (413) | 187 (634) | 508 (578) | 466 (710) | 421 (583) |
| 4-4 | – | 138 (138) | 132 (137) | 123 (133) | 108 (147) | 134 (138) |
| 5-3 | 2425 | 425 (628) | 416 (628) | 360 (702) | 267 (418) | 285 (318) |
| 5-4 | 2245 | 338 (347) | 343 (418) | 295 (378) | 337 (347) | 315 (344) |
| 6-2 | 3449 | 356 (448) | 295 (305) | 318 (448) | 686 (1278) | 296 (302) |
| 6-3 | 2665 | 411 (491) | 311 (393) | 299 (373) | 350 (480) | 375 (480) |
| 6-4 | 2244 | 237 (349) | 521 (561) | 233 (234) | 233 (235) | 232 (232) |
| 7-1 | 2857 | 409 (722) | 443 (1216) | 559 (882) | 524 (985) | 514 (817) |
| 7-2 | – | 485 (789) | 389 (786) | 810 (1235) | 487 (1109) | 682 (1904) |
| 7-3 | 3593 | 299 (796) | 167 (586) | 637 (1080) | 582 (891) | 1178 (1867) |
| 7-4 | – | 345 (424) | 341 (346) | 301 (358) | 340 (347) | 328 (422) |
| 8-1 | 6009 | 450 (1216) | 507 (938) | 401 (1074) | 461 (1072) | 256 (546) |
| 8-2 | 3449 | 317 (348) | 421 (1018) | 305 (1428) | 408 (1438) | 286 (346) |
| 8-3 | 3417 | 654 (2009) | 589 (1730) | 601 (1015) | 923 (1237) | 566 (1137) |
| 8-4 | – | 171 (178) | 167 (167) | 147 (147) | 238 (1159) | 174 (177) |

## Mỗi mạng đi được bao nhiêu phần màn (chỉ các màn đo được x của cờ, chỉ các màn mạng đó chưa học)

| Mạng | Số màn | Lượt chạm cờ | Trung bình của (x trung bình / x cờ) | Trung bình của (x xa nhất / x cờ) | Lớn nhất của (x xa nhất / x cờ) |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 17 | 0/340 | 13.5% | 21.2% | 59% |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 17 | 0/340 | 13.0% | 22.7% | 51% |
| mạng chuyên 1-1, bước 1.200.128 | 17 | 0/340 | 14.3% | 23.1% | 49% |
| mạng chuyên 4-1, bước 1.200.128 | 15 | 0/300 | 16.2% | 26.1% | 50% |
| mạng chuyên 2-3, bước 2.801.664 | 16 | 0/320 | 14.3% | 21.2% | 64% |

## So từng cặp: mạng 8 màn với từng mạng khác, trên các màn cả hai đều chưa học và đều đã chấm

| Mạng so sánh | Số màn chung | Mạng 8 màn xa hơn (x trung bình) | Hòa (chênh dưới 50) | Trung bình của x trung bình: 8 màn / mạng kia | Trung bình của x xa nhất: 8 màn / mạng kia |
|---|---|---|---|---|---|
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 24 | 11 | 9 | 378 / 332 | 631 / 596 |
| mạng chuyên 1-1, bước 1.200.128 | 24 | 6 | 12 | 378 / 412 | 631 / 703 |
| mạng chuyên 4-1, bước 1.200.128 | 21 | 4 | 9 | 378 / 453 | 641 / 835 |
| mạng chuyên 2-3, bước 2.801.664 | 23 | 7 | 11 | 383 / 407 | 612 / 658 |

## Màn 1-2 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 427 | 972 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 256 | 652 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 177 | 965 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | chưa đo | chưa đo | chưa đo | chưa đo | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 355 | 892 | – / – | 0/20 | chưa học màn này |

## Màn 1-3 (cờ ở x 2425)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 430 | 439 | 18% / 18% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 157 | 420 | 6% / 17% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 416 | 445 | 17% / 18% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 403 | 747 | 17% / 31% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 292 | 315 | 12% / 13% | 0/20 | chưa học màn này |

## Màn 1-4 (cờ ở x 2244)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 232 | 236 | 10% / 11% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 494 | 561 | 22% / 25% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 233 | 234 | 10% / 10% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | chưa đo | chưa đo | chưa đo | chưa đo | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 232 | 232 | 10% / 10% | 0/20 | chưa học màn này |

## Màn 2-2 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 874 | 1435 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 322 | 514 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 1096 | 1618 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 822 | 2163 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 1084 | 1522 | – / – | 0/20 | chưa học màn này |

## Màn 2-3 (cờ ở x 3593)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 259 | 1081 | 7% / 30% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 192 | 650 | 5% / 18% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 843 | 1762 | 23% / 49% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 1024 | 1792 | 28% / 50% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 3492 | 3593 | 97% / 100% | 18/20 | đã học màn này, không tính là màn lạ |

## Màn 2-4 (cờ ở x 2245)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 344 | 348 | 15% / 16% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 389 | 416 | 17% / 19% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 291 | 344 | 13% / 15% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 337 | 347 | 15% / 15% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 314 | 344 | 14% / 15% | 0/20 | chưa học màn này |

## Màn 3-3 (cờ ở x 2409)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 462 | 472 | 19% / 20% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 305 | 323 | 13% / 13% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 322 | 454 | 13% / 19% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | chưa đo | chưa đo | chưa đo | chưa đo | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 324 | 326 | 13% / 14% | 0/20 | chưa học màn này |

## Màn 3-4 (cờ ở x 2244)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 358 | 524 | 16% / 23% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 358 | 434 | 16% / 19% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 324 | 521 | 14% / 23% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 270 | 274 | 12% / 12% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 505 | 1435 | 22% / 64% | 0/20 | chưa học màn này |

## Màn 4-2 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 244 | 346 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 263 | 466 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 281 | 466 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 249 | 350 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 223 | 466 | – / – | 0/20 | chưa học màn này |

## Màn 4-3 (cờ ở x 2345)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 398 | 413 | 17% / 18% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 187 | 634 | 8% / 27% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 508 | 578 | 22% / 25% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 466 | 710 | 20% / 30% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 421 | 583 | 18% / 25% | 0/20 | chưa học màn này |

## Màn 4-4 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 138 | 138 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 132 | 137 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 123 | 133 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 108 | 147 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 134 | 138 | – / – | 0/20 | chưa học màn này |

## Màn 5-3 (cờ ở x 2425)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 425 | 628 | 18% / 26% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 416 | 628 | 17% / 26% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 360 | 702 | 15% / 29% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 267 | 418 | 11% / 17% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 285 | 318 | 12% / 13% | 0/20 | chưa học màn này |

## Màn 5-4 (cờ ở x 2245)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 338 | 347 | 15% / 15% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 343 | 418 | 15% / 19% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 295 | 378 | 13% / 17% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 337 | 347 | 15% / 15% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 315 | 344 | 14% / 15% | 0/20 | chưa học màn này |

## Màn 6-2 (cờ ở x 3449)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 356 | 448 | 10% / 13% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 295 | 305 | 9% / 9% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 318 | 448 | 9% / 13% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 686 | 1278 | 20% / 37% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 296 | 302 | 9% / 9% | 0/20 | chưa học màn này |

## Màn 6-3 (cờ ở x 2665)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 411 | 491 | 15% / 18% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 311 | 393 | 12% / 15% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 299 | 373 | 11% / 14% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 350 | 480 | 13% / 18% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 375 | 480 | 14% / 18% | 0/20 | chưa học màn này |

## Màn 6-4 (cờ ở x 2244)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 237 | 349 | 11% / 16% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 521 | 561 | 23% / 25% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 233 | 234 | 10% / 10% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 233 | 235 | 10% / 10% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 232 | 232 | 10% / 10% | 0/20 | chưa học màn này |

## Màn 7-1 (cờ ở x 2857)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 409 | 722 | 14% / 25% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 443 | 1216 | 16% / 43% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 559 | 882 | 20% / 31% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 524 | 985 | 18% / 34% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 514 | 817 | 18% / 29% | 0/20 | chưa học màn này |

## Màn 7-2 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 485 | 789 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 389 | 786 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 810 | 1235 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 487 | 1109 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 682 | 1904 | – / – | 0/20 | chưa học màn này |

## Màn 7-3 (cờ ở x 3593)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 299 | 796 | 8% / 22% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 167 | 586 | 5% / 16% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 637 | 1080 | 18% / 30% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 582 | 891 | 16% / 25% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 1178 | 1867 | 33% / 52% | 0/20 | chưa học màn này |

## Màn 7-4 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 345 | 424 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 341 | 346 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 301 | 358 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 340 | 347 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 328 | 422 | – / – | 0/20 | chưa học màn này |

## Màn 8-1 (cờ ở x 6009)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 450 | 1216 | 7% / 20% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 507 | 938 | 8% / 16% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 401 | 1074 | 7% / 18% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 461 | 1072 | 8% / 18% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 256 | 546 | 4% / 9% | 0/20 | chưa học màn này |

## Màn 8-2 (cờ ở x 3449)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 317 | 348 | 9% / 10% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 421 | 1018 | 12% / 30% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 305 | 1428 | 9% / 41% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 408 | 1438 | 12% / 42% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 286 | 346 | 8% / 10% | 0/20 | chưa học màn này |

## Màn 8-3 (cờ ở x 3417)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 654 | 2009 | 19% / 59% | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 589 | 1730 | 17% / 51% | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 601 | 1015 | 18% / 30% | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 923 | 1237 | 27% / 36% | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 566 | 1137 | 17% / 33% | 0/20 | chưa học màn này |

## Màn 8-4 (không đo được x của cờ)

| Mạng | x trung bình | x xa nhất | % màn (trung bình / xa nhất) | Lượt chạm cờ | Ghi chú |
|---|---|---|---|---|---|
| mạng 8 màn, bước 9.998.336 | 171 | 178 | – / – | 0/20 | chưa học màn này |
| mạng 2 màn (1-1 và 4-1), bước 2.904.064 | 167 | 167 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 1-1, bước 1.200.128 | 147 | 147 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 4-1, bước 1.200.128 | 238 | 1159 | – / – | 0/20 | chưa học màn này |
| mạng chuyên 2-3, bước 2.801.664 | 174 | 177 | – / – | 0/20 | chưa học màn này |
