from nicegui import ui

ui.dark_mode().enable()

with ui.column().classes("w-full items-center"):
    with ui.row(align_items="center"):
        ui.label("CAV Attack Panel").style('font-size: 48px; font-weight: bold; color: #ffffff;')
    with ui.row(align_items="center").style('padding-top: 48px;'):
        ui.button(text="Attack 1")
        ui.button(text="Attack 2")
        ui.button(text="Attack 3")



ui.run()
