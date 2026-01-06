# Fix  OpenMP library conflict coz multiple lib r loaded
export KMP_DUPLICATE_LIB_OK=TRUE

# Chainlit with watch mode
chainlit run app.py -w
