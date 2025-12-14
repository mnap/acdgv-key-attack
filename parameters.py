from types import SimpleNamespace

PARAMS = [
    SimpleNamespace(group="small", q=2, k=3, m=6, n=5, ell1=2, ell2=2),
    SimpleNamespace(group="small", q=3, k=4, m=7, n=7, ell1=2, ell2=2),
    SimpleNamespace(group="small", q=5, k=4, m=7, n=6, ell1=3, ell2=4),

    SimpleNamespace(group="acdgv_fig5", q=2, k=17, m=37, ell1=3, ell2=3, r=10, design=128),
    SimpleNamespace(group="acdgv_fig5", q=2, k=25, m=37, ell1=3, ell2=3, r= 6, design=128),
    SimpleNamespace(group="acdgv_fig5", q=2, k=35, m=43, ell1=2, ell2=2, r= 4, design=128),
    SimpleNamespace(group="acdgv_fig5", q=2, k=47, m=53, ell1=2, ell2=2, r= 3, design=128),
    SimpleNamespace(group="acdgv_fig5", q=2, k=51, m=59, ell1=2, ell2=2, r= 4, design=192),
    SimpleNamespace(group="acdgv_fig5", q= 2, k=23, m=47, ell1=3, ell2=3, r=12, design=256),
    SimpleNamespace(group="acdgv_fig5", q= 2, k=37, m=53, ell1=3, ell2=2, r= 8, design=256),
    SimpleNamespace(group="acdgv_fig5", q= 2, k=71, m=79, ell1=2, ell2=2, r= 4, design=256),

    SimpleNamespace(group="acdgv_fig6", q= 2, k=17, m=37, ell1=4, ell2=0, r=10, design=128),
    SimpleNamespace(group="acdgv_fig6", q=16, k=13, m=23, ell1=1, ell2=1, r= 5, design=128),
    SimpleNamespace(group="acdgv_fig6", q=16, k= 7, m=23, ell1=0, ell2=5, r= 8, design=128),
    SimpleNamespace(group="acdgv_fig6", q=2, k=23, m=43, ell1=5, ell2=0, r=10, design=192),
    SimpleNamespace(group="acdgv_fig6", q=2, k=33, m=47, ell1=5, ell2=0, r= 7, design=192),
    SimpleNamespace(group="acdgv_fig6", q=2, k=41, m=53, ell1=4, ell2=0, r= 6, design=192),
    SimpleNamespace(group="acdgv_fig6", q=16, k= 9, m=29, ell1=2, ell2=1, r=10, design=256),
    SimpleNamespace(group="acdgv_fig6", q=16, k=17, m=29, ell1=2, ell2=1, r= 8, design=256),
]


def get():
    return PARAMS
