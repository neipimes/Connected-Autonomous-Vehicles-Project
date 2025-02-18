from nicegui import ui

#ui.dark_mode().enable()

ui.colors()

ui.add_head_html('<style>body {background-color: #202020; }</style>')

with ui.column().classes("w-full items-center"):
    with ui.row(align_items="center"):
        ui.label("CAV Attack Panel").style('font-size: 48px; font-weight: bold; color: #ffffff;')
    with ui.row().style('padding-top: 12px;').classes('w-full justify-center'):
        with ui.column(align_items="center").classes('col'):
            ui.button(text="Attack 1").classes("w-full")
        with ui.column(align_items="center").classes('col'):
            ui.button(text="Attack 2").classes("w-full")
        with ui.column(align_items="center").classes('col'):
            ui.button(text="Attack 3").classes("w-full")



ui.run()
