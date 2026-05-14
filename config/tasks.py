TASKS = {
    'default': {
        'x': {
            'acg_id': [1, 7],
            'sp_id': [1, 7],
            'idl_id': [1, 7],
            'fas_id': [1, 7],
            'hip_id': [1, 7]
        },
        'y': {
            'bdi_sum': [0, 63],
            'suicid': [0, 12],
            'anger_state': [10, 40],
            'anger_trait': [10, 40],
            'anger_in': [8, 32],
            'anger_out': [9, 36],
            'anger_control': [7, 28],
            'emp_ec': [7, 35],
            'emp_pt': [7, 35],
            'emp_fs': [7, 35],
            'emp_pd': [7, 35],
            'ax_state': [20, 80],
            'ax_trait': [20, 80]
        },
        'models': [
            # {
            #     'layers': [16],
            #     'dropout': 0.2
            # },
            # {
            #     'layers': [32],
            #     'dropout': 0.2
            # },
            # {
            #     'layers': [64],
            #     'dropout': 0.2
            # },
            {
                'layers': [32, 32],
                'dropout': 0.2
            },
            # {
            #     'layers': [32, 32, 32],
            #     'dropout': 0.2
            # },
        ]
    },
    # 'grprepu': {
    #     'x': {
    #         'acg_id': [1, 7],
    #         'sp_id': [1, 7],
    #         'idl_id': [1, 7],
    #         'fas_id': [1, 7],
    #         'hip_id': [1, 7]
    #     },
    #     'y': {
    #         'grprepu1': [1, 7],
    #         'grprepu2': [1, 7]
    #     }
    # },
    # 'grpsp': {
    #     'x': {
    #         'acg_id': [1, 7],
    #         'sp_id': [1, 7],
    #         'idl_id': [1, 7],
    #         'fas_id': [1, 7],
    #         'hip_id': [1, 7]
    #     },
    #     'y': {
    #         'grpsp1': [1, 7],
    #         'grpsp2': [1, 7],
    #         'grpsp3': [1, 7],
    #         'grpsp4': [1, 8],
    #     }
    # }
}
