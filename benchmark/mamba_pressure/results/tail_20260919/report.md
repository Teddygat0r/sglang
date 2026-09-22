# Exploratory tail profiling

One repetition per mode: descriptive only, no confidence intervals. See TAIL_PROFILE.md for measurement limitations.

## eager_control

Mean/P95/P99 TTFT: 525.57/1586.00/1611.63 ms

## eager_profile

Mean/P95/P99 TTFT: 516.70/1567.79/1588.04 ms

```json
{
  "all": {
    "n": 192,
    "means": {
      "ttft_ms": 516.6975591273513,
      "cache_lookup_host_ms": 1.6157044947916666,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.29703784375,
      "restore_gpu_span_ms": 1.4327713300784428,
      "forward_host_ms": 31.762917104166668,
      "forward_gpu_span_ms": 292.4729694525401,
      "svd_host_overlap_ms": 234.90101619791668,
      "commit_loop_host_overlap_ms": 0.6881291666666667,
      "client_to_first_forward_ms": 238.48369878645832
    }
  },
  "tail": {
    "n": 11,
    "means": {
      "ttft_ms": 1578.6333285610783,
      "cache_lookup_host_ms": 0.069075,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.0,
      "restore_gpu_span_ms": 0,
      "forward_host_ms": 48.57148681818182,
      "forward_gpu_span_ms": 472.28782341697,
      "svd_host_overlap_ms": 646.4129545454546,
      "commit_loop_host_overlap_ms": 1.487405909090909,
      "client_to_first_forward_ms": 1136.070273909091
    }
  },
  "non_tail": {
    "n": 181,
    "means": {
      "ttft_ms": 452.1600261783402,
      "cache_lookup_host_ms": 1.7096985524861879,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.3758633480662983,
      "restore_gpu_span_ms": 1.5198458308014422,
      "forward_host_ms": 30.74140181767956,
      "forward_gpu_span_ms": 281.5449949022156,
      "svd_host_overlap_ms": 209.89200337016575,
      "commit_loop_host_overlap_ms": 0.6395543370165746,
      "client_to_first_forward_ms": 183.9342384198895
    }
  },
  "compressed": {
    "n": 126,
    "means": {
      "ttft_ms": 150.3881732563651,
      "cache_lookup_host_ms": 2.4194292857142856,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.976438619047619,
      "restore_gpu_span_ms": 2.18327059821477,
      "forward_host_ms": 23.14812973015873,
      "forward_gpu_span_ms": 103.59396961757115,
      "svd_host_overlap_ms": 112.71682732539682,
      "commit_loop_host_overlap_ms": 0.5579801825396825,
      "client_to_first_forward_ms": 72.4815578015873
    }
  },
  "full": {
    "n": 0,
    "means": {}
  },
  "miss": {
    "n": 66,
    "means": {
      "ttft_ms": 1216.015477608325,
      "cache_lookup_host_ms": 0.08132080303030303,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.0,
      "restore_gpu_span_ms": 0,
      "forward_host_ms": 48.209329363636364,
      "forward_gpu_span_ms": 653.0601509556626,
      "svd_host_overlap_ms": 468.1617404090909,
      "commit_loop_host_overlap_ms": 0.9365954090909091,
      "client_to_first_forward_ms": 555.396877030303
    }
  }
}

```

## off_control

Mean/P95/P99 TTFT: 1055.48/1519.77/1524.25 ms

## off_profile

Mean/P95/P99 TTFT: 1055.59/1518.81/1527.99 ms

```json
{
  "all": {
    "n": 192,
    "means": {
      "ttft_ms": 1055.5911247308056,
      "cache_lookup_host_ms": 0.332585796875,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.21139109375,
      "restore_gpu_span_ms": 0.277341832096378,
      "forward_host_ms": 43.384832755208336,
      "forward_gpu_span_ms": 576.8643951614698,
      "svd_host_overlap_ms": 0.0,
      "commit_loop_host_overlap_ms": 0.0092415,
      "client_to_first_forward_ms": 478.07274097916667
    }
  },
  "tail": {
    "n": 11,
    "means": {
      "ttft_ms": 1523.6869479783556,
      "cache_lookup_host_ms": 0.06078836363636363,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.0,
      "restore_gpu_span_ms": 0,
      "forward_host_ms": 46.63394454545455,
      "forward_gpu_span_ms": 451.11362214521927,
      "svd_host_overlap_ms": 0.0,
      "commit_loop_host_overlap_ms": 0.010349090909090908,
      "client_to_first_forward_ms": 1101.7404032727272
    }
  },
  "non_tail": {
    "n": 181,
    "means": {
      "ttft_ms": 1027.1433122682472,
      "cache_lookup_host_ms": 0.3491038729281768,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.22423806629834253,
      "restore_gpu_span_ms": 0.29419686056632366,
      "forward_host_ms": 43.18737292265193,
      "forward_gpu_span_ms": 584.5067073337281,
      "svd_host_overlap_ms": 0.0,
      "commit_loop_host_overlap_ms": 0.009174187845303867,
      "client_to_first_forward_ms": 440.1702863646409
    }
  },
  "compressed": {
    "n": 0,
    "means": {}
  },
  "full": {
    "n": 26,
    "means": {
      "ttft_ms": 462.00938501323645,
      "cache_lookup_host_ms": 2.0411606153846154,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.561041923076923,
      "restore_gpu_span_ms": 2.0480627600963297,
      "forward_host_ms": 22.397275269230768,
      "forward_gpu_span_ms": 203.56014427771936,
      "svd_host_overlap_ms": 0.0,
      "commit_loop_host_overlap_ms": 0.005377846153846154,
      "client_to_first_forward_ms": 262.4347267307692
    }
  },
  "miss": {
    "n": 166,
    "means": {
      "ttft_ms": 1148.561758662473,
      "cache_lookup_host_ms": 0.06497769277108434,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.0,
      "restore_gpu_span_ms": 0,
      "forward_host_ms": 46.67204055421687,
      "forward_gpu_span_ms": 635.3337356613343,
      "svd_host_overlap_ms": 0.0,
      "commit_loop_host_overlap_ms": 0.009846650602409638,
      "client_to_first_forward_ms": 511.8473697168675
    }
  }
}

```

## pressure_control

Mean/P95/P99 TTFT: 509.91/1611.69/1616.49 ms

## pressure_profile

Mean/P95/P99 TTFT: 508.14/1588.69/1606.27 ms

```json
{
  "all": {
    "n": 192,
    "means": {
      "ttft_ms": 508.1398988550063,
      "cache_lookup_host_ms": 1.48239296875,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.1230500260416667,
      "restore_gpu_span_ms": 1.3027181718498468,
      "forward_host_ms": 32.403248895833336,
      "forward_gpu_span_ms": 291.9625676671664,
      "svd_host_overlap_ms": 86.96750473958333,
      "commit_loop_host_overlap_ms": 0.20917296875,
      "client_to_first_forward_ms": 234.19173768229166
    }
  },
  "tail": {
    "n": 11,
    "means": {
      "ttft_ms": 1598.7639601596377,
      "cache_lookup_host_ms": 0.11031736363636364,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.0,
      "restore_gpu_span_ms": 0,
      "forward_host_ms": 52.37326590909091,
      "forward_gpu_span_ms": 486.3880792097612,
      "svd_host_overlap_ms": 484.95150072727273,
      "commit_loop_host_overlap_ms": 1.2814821818181819,
      "client_to_first_forward_ms": 1138.1538867272727
    }
  },
  "non_tail": {
    "n": 181,
    "means": {
      "ttft_ms": 441.85887855472487,
      "cache_lookup_host_ms": 1.5657787790055249,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.191301685082873,
      "restore_gpu_span_ms": 1.3818888894760806,
      "forward_host_ms": 31.189601453038673,
      "forward_gpu_span_ms": 280.14665260104186,
      "svd_host_overlap_ms": 62.78063205524862,
      "commit_loop_host_overlap_ms": 0.14400500552486187,
      "client_to_first_forward_ms": 179.25481149723757
    }
  },
  "compressed": {
    "n": 126,
    "means": {
      "ttft_ms": 138.6684783187414,
      "cache_lookup_host_ms": 2.2069022698412697,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 1.7113143253968255,
      "restore_gpu_span_ms": 1.9850943571045285,
      "forward_host_ms": 22.537090428571428,
      "forward_gpu_span_ms": 100.42331235370939,
      "svd_host_overlap_ms": 9.927207325396825,
      "commit_loop_host_overlap_ms": 0.05732723015873016,
      "client_to_first_forward_ms": 70.39492657936508
    }
  },
  "full": {
    "n": 0,
    "means": {}
  },
  "miss": {
    "n": 66,
    "means": {
      "ttft_ms": 1213.494428969694,
      "cache_lookup_host_ms": 0.09923884848484849,
      "cache_lookup_gpu_span_ms": 0,
      "restore_host_ms": 0.0,
      "restore_gpu_span_ms": 0,
      "forward_host_ms": 51.23864233333333,
      "forward_gpu_span_ms": 657.6284187201297,
      "svd_host_overlap_ms": 234.04443616666666,
      "commit_loop_host_overlap_ms": 0.49906028787878787,
      "client_to_first_forward_ms": 546.8947406969697
    }
  }
}

```
