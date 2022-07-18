# set up the ecosystem with bacterified macroecology names
ecosystem = {
    'antilopis': {
        'antilopis': -1,
        'baboonis': 0,
        'bisonis': -2,
        'buffalonis': -2,
        'cheetahis': -.1,
        'duckis': 0,
        'eagleis': 1,
        'lionis': -.2,
        'tigris': -.2,
        'zebrais': -2
    },
    'baboonis': {
        'antilopis': 0,
        'baboonis': -1,
        'bisonis': 0,
        'buffalonis': 0,
        'cheetahis': -.5,
        'duckis': 0,
        'eagleis': -1,
        'lionis': -.5,
        'tigris': -.5,
        'zebrais': 0
    },
    'bisonis': {
        'antilopis': -.5,
        'baboonis': 0,
        'bisonis': -.5,
        'buffalonis': -2,
        'cheetahis': -.5,
        'duckis': 0,
        'eagleis': 1,
        'lionis': -1,
        'tigris': -1,
        'zebrais': -2
    },
    'buffalonis': {
        'antilopis': -.5,
        'baboonis': 0,
        'bisonis': -2,
        'buffalonis': -.5,
        'cheetahis': -.5,
        'duckis': 0,
        'eagleis': 1,
        'lionis': -1,
        'tigris': -1,
        'zebrais': -2
    },
    'cheetahis': {
        'antilopis': .5,
        'baboonis': .125,
        'bisonis': 1.25,
        'buffalonis': 1.5,
        'cheetahis': -2.5,
        'duckis': .15,
        'eagleis': 0,
        'lionis': 0,
        'tigris': 0,
        'zebrais': 1.5
    },
    'duckis': {
        'antilopis': -.01,
        'baboonis': 0,
        'bisonis': -.02,
        'buffalonis': -.02,
        'cheetahis': -.01,
        'duckis': -5,
        'eagleis': 0,
        'lionis': -.15,
        'tigris': -.15,
        'zebrais': -.02
    },
    'eagleis': {
        'antilopis': 0.01,
        'baboonis': .5,
        'bisonis': 0,
        'buffalonis': 0,
        'cheetahis': 0,
        'duckis': 0,
        'eagleis': -1,
        'lionis': 0,
        'tigris': 0,
        'zebrais': 0
    },
    'lionis': {
        'antilopis': 1,
        'baboonis': .25,
        'bisonis': 2.5,
        'buffalonis': 3,
        'cheetahis': 0,
        'duckis': .15,
        'eagleis': 0,
        'lionis': -1,
        'tigris': -15,
        'zebrais': 3
    },
    'tigris': {
        'antilopis': 1,
        'baboonis': .25,
        'bisonis': 2.5,
        'buffalonis': 3,
        'cheetahis': 0,
        'duckis': .15,
        'eagleis': 0,
        'lionis': -15,
        'tigris': -1,
        'zebrais': 3
    },
    'zebrais': {
        'antilopis': -.25,
        'baboonis': 0,
        'bisonis': -1,
        'buffalonis': -1,
        'cheetahis': -.5,
        'duckis': -.1,
        'eagleis': 0,
        'lionis': -1,
        'tigris': -1,
        'zebrais': -.5,
    }
}

growth_rates = {
    'antilopis': .85,
    'baboonis': 1,
    'bisonis': .6,
    'buffalonis': .6,
    'cheetahis': -.75,
    'duckis': .3,
    'eagleis': .8,
    'lionis': -1.5,
    'tigris': -1.5,
    'zebrais': .65
}

species_styles = {
    'antilopis': {
        'c': '#ceb301',
        's': '-',
        'w': 1
    },
    'baboonis': {
        'c': '#a00498',
        's': '-',
        'w': 1
    },
    'bisonis': {
        'c': '#88b378',
        's': '-',
        'w': 3
    },
    'buffalonis': {
        'c': '#88b378',
        's': '--',
        'w': 3
    },
    'cheetahis': {
        'c': '#d9544d',
        's': ':',
        'w': 2
    },
    'duckis': {
        'c': '#a8a495',
        's': '-',
        'w': 1
    },
    'eagleis': {
        'c': '#9dbcd4',
        's': '-',
        'w': 1
    },
    'lionis': {
        'c': '#ef4026',
        's': '-',
        'w': 3
    },
    'tigris': {
        'c': '#ef4026',
        's': '--',
        'w': 3
    },
    'zebrais': {
        'c': '#070d0d',
        's': '-',
        'w': 1
    }
}
