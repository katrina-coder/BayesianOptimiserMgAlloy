import ipywidgets as widgets
from IPython.display import HTML
from ipywidgets import Layout, HBox, VBox
from IPython.display import display, clear_output
if 'google.colab' in str(get_ipython()):
    from BayesianOptimiserMgAlloy.optimiser import *
else:
    from optimiser import *


def extractSettingsFromGUI(GUI_inputs, mode):
    settings = scanSettings(mode)
    uppers = []
    for key in settings.range_based_inputs: 
        if key not in ['Extruded', 'ECAP','Cast_Slow', 'Cast_Fast', 'Cast_HT','Wrought']:
            settings.range_based_inputs[key] = [GUI_inputs['range_based_inputs'][key][0].value,
                                                GUI_inputs['range_based_inputs'][key][1].value] 
        else:
            upper = [1 if GUI_inputs["bo_settings"]["Heat Treatment"][key].value=='True' else 0][0]
            uppers.append(upper)
            settings.range_based_inputs[key] = [0, upper]
    if 1 not in uppers:
        settings.range_based_inputs['Extruded']=[0,1]
                                            
    settings.sampling_size = int(GUI_inputs["bo_settings"]["Sampling Size"].value)
    settings.num_of_suggestions = int(GUI_inputs['bo_settings']['Number of Suggestions'].value)
    settings.num_elems = int(GUI_inputs['bo_settings']['Number of Elements'].value) 
    settings.sum_elems = int(GUI_inputs['bo_settings']['Percentage Sum of Elements'].value) 
    settings.normalize_target = [True if GUI_inputs["bo_settings"]["Normalize Target"].value=='Yes' else False][0]
    settings.append_suggestion = [True if GUI_inputs["bo_settings"]["Append Suggestion"].value=='Yes' else False][0] 
    output_names = []
    if GUI_inputs["bo_settings"]["Output Names"].value == 'UTS only':
        output_names = ['UTS']
    elif GUI_inputs["bo_settings"]["Output Names"].value == 'Ductility only':
        output_names = ['Ductility']
    else:
        output_names = ['UTS', 'Ductility']
    settings.output_names = output_names
    return settings

def generateModeSelectionGUI(mode = 'Bayesian Optimisation'):
    mode_dropdown = widgets.Dropdown(
        options=['Bayesian Optimisation'],
        value=mode,
        description='<b>Select Mode:</b>',
        style={'description_width': 'initial'},
        disabled=False
    )
    display(mode_dropdown)
    generateMainGUI(mode)
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            clear_output(wait=True)
            generateModeSelectionGUI(change['new'])
    mode_dropdown.observe(on_change)

def generateMainGUI(mode):
    settings = scanSettings(mode)
    KEY_LABEL_WIDTH = "40px"
    TO_LABEL_WIDTH = "15px"
    INPUT_BOX_WIDTH = "60px"
    INPUT_BOX_HEIGHT = "20px"

    LEFT_RIGHT_PADDING = Layout(margin="0px 30px 0px 30px")
    BOTTOM_PADDING = Layout(margin="0px 0px 5px 0px")

    default_input_box_layout = Layout(width=INPUT_BOX_WIDTH, height=INPUT_BOX_HEIGHT)

    if mode=='Bayesian Optimisation':
        GUI_inputs = {"range_based_inputs": {},
                    # "constant_inputs": {},
                    # "categorical_inputs": {},
                    "bo_settings": {}
                    }
    
        range_based_inputs_VBox = [widgets.HTML("<b>Compositional range (wt. %) </b>")]
        for key in settings.range_based_inputs:
            if key not in ['Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']:
                key_label = widgets.Label(f"{key}:", layout=Layout(width=KEY_LABEL_WIDTH))
                lower_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][0], layout=default_input_box_layout)
                to_label = widgets.Label("to", layout=Layout(width=TO_LABEL_WIDTH))
                upper_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][1], layout=default_input_box_layout)
                range_based_inputs_VBox.append(HBox([key_label, lower_bound_box, to_label, upper_bound_box]))
                GUI_inputs["range_based_inputs"][key] = [lower_bound_box, upper_bound_box]
                    
        
        ht_settings_VBox = [widgets.HTML("<b>Thermomechanical process</b>")]
        GUI_inputs["bo_settings"]["Heat Treatment"] = {}
        ht_names = ['Extruded', 'ECAP', 'Cast (slow cool)', 'Cast (fast cool)', 'Cast and Heat-treated',  'Wrought']
        ht_keys = ['Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']
        for i in range(len(ht_keys)):
            key_label = widgets.Label(f"{ht_names[i]}:", layout=Layout(width='180px'))
            input_box = widgets.RadioButtons(value=settings.HT, options=['True', 'False'], description = '', disabled=False, indent=False)
            ht_settings_VBox.append(HBox([key_label, input_box]))
            GUI_inputs["bo_settings"]["Heat Treatment"][ht_keys[i]] = input_box 
        
        bo_settings_width = '270px'
        scan_settings_VBox = [widgets.HTML("<b>Bayesian-Optimisation settings</b>")]
        label = widgets.Label("Sample size: ",layout=Layout(width=bo_settings_width))
        input_box = widgets.FloatText(value=settings.sampling_size, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Sampling Size"] = input_box
        
        label = widgets.Label("Number of suggestions: ",layout=Layout(width=bo_settings_width))
        input_box = widgets.FloatText(value=settings.num_of_suggestions, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Number of Suggestions"] = input_box
        
        label = widgets.Label("Max. number of alloying elements: ",layout=Layout(width=bo_settings_width))
        input_box = widgets.FloatText(value=settings.num_elems, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Number of Elements"] = input_box
        
        label = widgets.Label("Max. sum of alloying elements (wt. %): ",layout=Layout(width=bo_settings_width))
        input_box = widgets.FloatText(value=settings.sum_elems, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Percentage Sum of Elements"] = input_box
        
        label = widgets.Label("Normalise target: ",layout=Layout(width=bo_settings_width))
        input_box = widgets.RadioButtons(value=settings.normalize_target, options=['Yes', 'No'], description = '', disabled=False, indent=False)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Normalize Target"] = input_box 
        
        label = widgets.Label("Append suggestion in iterations: ",layout=Layout(width=bo_settings_width))
        input_box = widgets.RadioButtons(value=settings.append_suggestion, options=['Yes', 'No'], description = '', disabled=False, indent=False)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Append Suggestion"] = input_box 
        
        
        label = widgets.Label("Output designed alloys to be optimised for: ",layout=Layout(width=bo_settings_width))
        input_box = widgets.RadioButtons(value='Both UTS and ductility', options=['UTS only', 'Ductility only', 'Both UTS and ductility'], description = '', disabled=False, indent=False)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Output Names"] = input_box
        first_column = VBox(range_based_inputs_VBox)
    
        second_column = VBox([VBox(ht_settings_VBox),
                            # VBox(categorical_inputs_VBox, layout=BOTTOM_PADDING),
                            VBox(scan_settings_VBox)], layout=LEFT_RIGHT_PADDING)
        
        display(HTML('<link rel="stylesheet" href="//stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"/>'))
        display(HBox([first_column, second_column]))
    
        run_scan_button = widgets.Button(
                                         #description="Run Optimiser", 
                                         icon='play')
        display(run_scan_button)
        
        def on_button_clicked(b):
            print('========== Bayesian Optimisation Started ==========')
            optimiser(extractSettingsFromGUI(GUI_inputs, mode))
    
        run_scan_button.on_click(on_button_clicked)
    
    else:
        GUI_inputs = {"range_based_inputs": {},
                    # "constant_inputs": {},
                    # "categorical_inputs": {},
                    "bo_settings": {}
                    }
    
        range_based_inputs_VBox = [widgets.HTML("<b>Compositional range (wt. %) </b>")]
        for key in settings.range_based_inputs:
            key_label = widgets.Label(f"{key}:", layout=Layout(width=KEY_LABEL_WIDTH))
            lower_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][0], layout=default_input_box_layout)
            to_label = widgets.Label("to", layout=Layout(width=TO_LABEL_WIDTH))
            upper_bound_box = widgets.FloatText(value=settings.range_based_inputs[key][1], layout=default_input_box_layout)
            range_based_inputs_VBox.append(HBox([key_label, lower_bound_box, to_label, upper_bound_box]))
            GUI_inputs["range_based_inputs"][key] = [lower_bound_box, upper_bound_box]
    
        # if bool(settings.constant_inputs):
        if False:
            constant_inputs_VBox = [widgets.HTML("<b>Constant Inputs</b>")]
            for key in settings.constant_inputs:
                key_label = widgets.Label(f"{key}:", layout=Layout(width=KEY_LABEL_WIDTH * 10))
                value_box = widgets.FloatText(value=settings.constant_inputs[key], layout=default_input_box_layout)
                constant_inputs_VBox.append(HBox([key_label, value_box]))
                GUI_inputs["constant_inputs"][key] = value_box
    
        scan_settings_VBox = [widgets.HTML("<b>Bayesian-Optimisation Settings</b>")]
        label = widgets.Label("Sample Size: ")
        input_box = widgets.FloatText(value=settings.sampling_size, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Sampling Size"] = input_box
        
        label = widgets.Label("Number of Suggestions: ")
        input_box = widgets.FloatText(value=settings.num_of_suggestions, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Number of Suggestions"] = input_box
        
        label = widgets.Label("Max. number of alloying elements: ")
        input_box = widgets.FloatText(value=settings.num_elems, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Number of Elements"] = input_box
        
        label = widgets.Label("Max. sum of alloying elements (wt. %): ")
        input_box = widgets.FloatText(value=settings.sum_elems, layout=default_input_box_layout)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Percentage Sum of Elements"] = input_box
        
        label = widgets.Label("Normalise Target: ")
        input_box = widgets.RadioButtons(value=settings.normalize_target, options=['Yes', 'No'], description = '', disabled=False, indent=False)
        scan_settings_VBox.append(HBox([label, input_box]))
        GUI_inputs["bo_settings"]["Normalize Target"] = [True if input_box.value=='True' else False][0]
        
        label = widgets.Label("Output: ")
        input_box = widgets.RadioButtons(value='UTS only', options=['UTS only', 'Ductility only', 'Both UTS and ductility'], description = '', disabled=False, indent=False)
        scan_settings_VBox.append(HBox([label, input_box]))
        output_names = []
        if input_box.value == 'UTS only':
            output_names = ['UTS']
        elif input_box.value == 'Ductility only':
            output_names = ['Ductility']
        else:
            output_names = ['UTS', 'Ductility']
        GUI_inputs["bo_settings"]["Output Names"] = output_names
        
        ## label = widgets.Label("Output Names: ")
        ## uts_value = ['UTS'] in settings.output_names
        ## input_box_1 = widgets.Checkbox(value=uts_value, description='UTS', disabled=False)
        ## duct_value = ['Ductility'] in settings.output_names
        ## input_box_2 = widgets.Checkbox(value=duct_value, description='Ductility', disabled=False)
        ## scan_settings_VBox.append(HBox([label]))
        ## scan_settings_VBox.append(HBox([input_box_1]))
        ## scan_settings_VBox.append(HBox([input_box_2]))
        ## output_names = []
        ## if input_box_1:
        ##     output_names.append('UTS')
        ## if input_box_2:
        ##     output_names.append('Ductility')
        ## print(input_box_1.value)
        ## GUI_inputs["bo_settings"]["Output Names"] = output_names
        
        
        
        # for key in settings.targets:
        #     input_box = widgets.FloatText(value=settings.targets[key], layout=default_input_box_layout)
        #     row = HBox([widgets.HTML(f'Target {key}:'), input_box])
        #     scan_settings_VBox.append(row)
        #     GUI_inputs["scan_settings"][key] = input_box
    
        first_column = VBox(range_based_inputs_VBox)
        second_column = VBox([
                            # VBox(categorical_inputs_VBox, layout=BOTTOM_PADDING),
                            VBox(scan_settings_VBox)], layout=LEFT_RIGHT_PADDING)
        display(HBox([first_column, second_column]))
        
        display(HTML('<link rel="stylesheet" href="//stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"/>'))
        run_scan_button = widgets.Button(
                                         #description="Run Optimiser", 
                                         icon='play')
        display(run_scan_button)
        
        def on_button_clicked(b):
            print('========== Bayesian Optimisation Started ==========')
            optimiser(extractSettingsFromGUI(GUI_inputs, mode))
    
        run_scan_button.on_click(on_button_clicked)
    
