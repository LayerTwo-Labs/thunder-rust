- Submission Notes

* **Reduced chunk size** in authorization verification from 16384 to 8192 for better parallelization efficiency
* **Parallelized merkle leaf computation** using rayon's `par_iter()` with chunked processing (50 transactions per chunk) instead of sequential computation
* **Added parallel hash computation** for large coinbase data (>100 items) using Blake3's parallel capabilities
* **Optimized memory access patterns** by processing transactions in smaller, cache-friendly chunks rather than one large sequential operation

I was also implementing the optoimizations for utrexxo, but I felt sort of time see branch LayerTwo-Labs/thunder-rust@master...Gmin2:thunder-rust:rustrexxo