from nicegui import ui

## Icons
# Engine override: https://www.flaticon.com/free-icons/automotive + https://www.flaticon.com/free-icons/thunder
# Camera DDOS: https://www.flaticon.com/free-icons/camera + https://www.flaticon.com/free-icons/no-entry
# Ultrasonic disruption: https://www.flaticon.com/free-icons/top-down-car + https://www.flaticon.com/free-icons/radar + https://www.flaticon.com/free-icons/close

#ui.dark_mode().enable()

ui.colors()

ui.add_head_html('<style>body {background-color: #202020; }</style>')

with ui.column().classes("w-full items-center"):
    with ui.row(align_items="center"):
        ui.label("CAV Attack Panel").style('font-size: 54px; font-weight: bold; color: #ffffff;')
    with ui.row().style('padding-top: 12px;').classes('w-full justify-center'):
        with ui.column(align_items="center").classes('col'):
            with ui.button().classes("w-full ma-10"):
                ui.image("img\\override-motor.png").classes("flex w-95 h-95")
                ui.label("Override Motor").style("font-size: 40px").classes("pt-5 pb-5")
        with ui.column(align_items="center").classes('col'):
            with ui.button().classes("w-full"):
                ui.image("img\\no-camera.png").classes("flex w-95 h-95")
                ui.label("Disable Camera").style("font-size: 40px").classes("pt-5 pb-5")
        with ui.column(align_items="center").classes('col'):
            with ui.button().classes("w-full"):
                ui.image("img\\no-ultrasonic.png").classes("flex w-95 h-95")
                ui.label("Disrupt Ultrasonic").style("font-size: 40px").classes("pt-5 pb-5")



ui.run()
