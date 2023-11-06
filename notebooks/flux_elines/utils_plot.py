from matplotlib.offsetbox import AnchoredText

def textonly(ax, txt, fontsize=14, loc=3, fontweight='bold', *args, **kwargs):
    at = AnchoredText(txt,
                      prop=dict(size=fontsize, fontweight=fontweight),
                      frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at


def textonly2(ax, txt, fontsize=14, loc=3, fontweight='bold', *args, **kwargs):
    at = AnchoredText(txt,
                      prop=dict(size=fontsize, fontweight=fontweight),
                      frameon=False,
                      loc=loc)
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at
