import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(mo):
    get_state_sl, set_state_sl = mo.state(-1)
    get_state_old_df, set_state_old_df = mo.state(None)
    #get_state_old_df_view, set_state_old_df_view = mo.state(None)
    get_state_view_changed, set_state_view_changed  = mo.state(False)
    nb_path = mo.notebook_location().joinpath('./Rezepte/')
    return (
        get_state_old_df,
        get_state_sl,
        get_state_view_changed,
        nb_path,
        set_state_old_df,
        set_state_sl,
        set_state_view_changed,
    )


@app.cell
def _(mo):
    mo.md("""## Hauptmenü erstellen""")
    return


@app.cell
def _(mo):
    cooking_dropdown_dict = mo.ui.dropdown(options={"Hauptgericht":1, "Nachspeise":2, "Vorspeise":3, "alles": 4},
                            value="Hauptgericht", # initial value
                            label="Rezept für ")

    baking_dropdown_dict = mo.ui.dropdown(options={"Herzhaft":1, "Suess":2, "Sonstiges":3, "alles": 4},
                            value="Herzhaft", # initial value
                            label="Rezept für ")
    shopping_list_mode_switch = mo.ui.switch(label='edit mode', value=False)

    food_tabs = mo.ui.tabs(
        {
            "Speiseplan": "",
            "Einkaufsliste": shopping_list_mode_switch,
            "Kochen": cooking_dropdown_dict,
            "Backen": baking_dropdown_dict,
            "Sonstiges": "",
        }
    )
    return (
        baking_dropdown_dict,
        cooking_dropdown_dict,
        food_tabs,
        shopping_list_mode_switch,
    )


@app.cell
def _(food_tabs):
    food_tabs
    return


@app.cell
def _(show_it):
    show_it
    return


@app.cell
def _(mo):
    mo.md("""## Wenn der Tab geändert wird, Daten neu laden.""")
    return


@app.cell
def _(apply_filter, food_tabs, get_data, nb_path, pd):
    food_tabs.value
    df = get_data()   # Lade Daten
    filtered_df = apply_filter(df)    # Filtere die zu zeigenden Daten ohne Metainformationen
    df_mp = pd.read_csv(str(nb_path / "Speiseplan.csv"), sep = ';', encoding='utf8')
    on_mp_idx = df_mp['idx_rezept'].to_list()
    return df, df_mp, filtered_df, on_mp_idx


@app.cell
def _(
    baking_dropdown_dict,
    cooking_dropdown_dict,
    food_tabs,
    nb_path,
    pd,
    set_state_old_df,
):
    ##### Lade Daten aus csv-Dateien ##############

    def get_data():
        _base_dataframe_dict = {
            "Kochen": nb_path.joinpath('Rezepte.csv'),
            "Backen": nb_path.joinpath('Rezepte.csv'),
            "Sonstiges": nb_path.joinpath('Rezepte.csv'),
            "Einkaufsliste": nb_path.joinpath('Einkaufsliste.csv'),
            "Speiseplan": nb_path.joinpath('Speiseplan.csv')
        }
        _file = _base_dataframe_dict[food_tabs.value]
        _df = pd.read_csv(_file, sep = ';', encoding = 'utf8')
        return _df



    #### Standard Filterung: was wird angezeigt und was nicht...
    def apply_filter(_df):
        _act_view_dict = {
            "Kochen": cooking_dropdown_dict,
            "Backen": baking_dropdown_dict
        }
        if food_tabs.value in ['Kochen', 'Backen', 'Sonstiges']:
            if _act_view_dict[food_tabs.value].selected_key == 'alles':
                _filtered_df = _df[_df['kind'] == food_tabs.value]
            else:
                _filtered_df = _df[(_df['sub_kind_1'] == _act_view_dict[food_tabs.value].selected_key) & (_df['kind'] == food_tabs.value)]

            _filtered_df_view = _filtered_df[['name', 'comments', 'hashtags', 'days_passed','ingredients']].reset_index(drop=True)

        elif food_tabs.value == 'Einkaufsliste':
            set_state_old_df(_df.copy(deep=True))
            _filtered_df_view = _df
        elif food_tabs.value == 'Speiseplan':
            _filtered_df_view = _df

        return _filtered_df_view
    return apply_filter, get_data


@app.cell
def _(mo):
    mo.md("""## Hier wird gesteuert, was angezeigt wird über Definition von show_it""")
    return


@app.cell
def _(
    filtered_df,
    food_tabs,
    get_state_old_df,
    get_state_view_changed,
    ing_table,
    mo,
    option_header,
    set_state_sl,
    shopping_list_mode_switch,
    sl_add_amount,
    sl_add_item,
    sl_add_rank,
    sl_add_submit_button,
    sl_add_unit,
    table,
    update_button,
    update_sl_df,
):
    get_state_view_changed()

    ###
    if food_tabs.value == 'Einkaufsliste':
        if shopping_list_mode_switch.value == False:  # in shopping mode add remove buttons for the view
            filtered_df_sl = get_state_old_df().copy()
            _loc_butt = mo.ui.array([
                mo.ui.run_button(label = '-', on_change=lambda v, i=i: set_state_sl(i)) for i in filtered_df_sl.index.to_list()])
            filtered_df_sl['remove'] = list(_loc_butt)
        elif shopping_list_mode_switch.value == True: # in edit mode show every entry as ui-element
            filtered_df_sl = get_state_old_df().copy()
            _article = mo.ui.array(
            [mo.ui.text(label = None, value = n, on_change=lambda v, i=i: update_sl_df(0,i,v)) for n, i in zip(filtered_df_sl['artikel'], filtered_df_sl.index.to_list())]
        )
            _numbers = mo.ui.array(
            [mo.ui.number(label = "", start = 0, value=n, on_change=lambda v, i=i: update_sl_df(1,i,v)) for n, i in zip(filtered_df_sl['menge'], filtered_df_sl.index.to_list()) ]
        )
            _unit = mo.ui.array(
            [mo.ui.text(label=None, value = n, on_change=lambda v, i=i: update_sl_df(2,i,v)) for n, i in zip(filtered_df_sl['einheit'], filtered_df_sl.index.to_list())]

        )
            _rang = mo.ui.array(
            [mo.ui.number(label = "", start = 0, value = n, on_change=lambda v, i=i: update_sl_df(3,i,v)) for n, i in zip(filtered_df_sl['rang'], filtered_df_sl.index.to_list()) ]
        )
            _loc_butt = mo.ui.array([
                mo.ui.run_button(label = '-', on_change=lambda v, i=i: set_state_sl(i)) for i in filtered_df_sl.index.to_list()])
            filtered_df_sl['artikel'] = list(_article)
            filtered_df_sl['remove'] = list(_loc_butt)
            filtered_df_sl['menge'] = list(_numbers)
            filtered_df_sl['einheit'] = list(_unit)
            filtered_df_sl['rang'] = list(_rang)
            _arr = mo.hstack([sl_add_item, sl_add_amount, sl_add_unit, sl_add_rank, sl_add_submit_button])

        act_tab = mo.ui.table(filtered_df_sl.to_dict('list'), selection = None, show_download=False, pagination=False)



    if food_tabs.value in ['Backen', 'Kochen', 'Sonstiges']: # Wenn Rezept ausgewählt...
        show_recipe = lambda : mo.vstack([option_header, table]) if 'ing_table' not in globals() else mo.vstack([option_header, ing_table, table])
    elif food_tabs.value == 'Einkaufsliste':   # Wenn Einkaufsliste ausgewählt
        show_recipe = lambda x = shopping_list_mode_switch.value: mo.vstack([act_tab, update_button.right()]) if x == False else mo.vstack([_arr, act_tab, update_button.right()])
    elif food_tabs.value == 'Speiseplan':
        show_recipe = lambda: mo.ui.table(filtered_df, show_download=False, pagination=False, selection = None, show_column_summaries=False)


    show_it = show_recipe()
    return act_tab, filtered_df_sl, show_it, show_recipe


@app.cell
def _(mo):
    mo.md("""## Hier wird table definiert - eventuell splitten je nach aktiviertem Tab...""")
    return


@app.cell
def _(filtered_df, food_tabs, mo):
    if food_tabs.value in ['Kochen', 'Backen', 'Sonstiges']:
        table = mo.ui.table(filtered_df, selection = 'single', show_download=False, show_column_summaries=False)
    elif food_tabs.value == 'Einkaufsliste':
        table = mo.ui.table(filtered_df, selection = 'single', show_download=False, show_column_summaries=False)
    return (table,)


@app.cell
def _(mo):
    mo.md("""# Hier kommen Funktionen und Definitionen für Rezepte""")
    return


@app.cell
def _(mo):
    mo.md("""### Definition des Optionen-Headers wenn ein Rezept ausgewählt wird""")
    return


@app.cell
def _(add_meal_to_mp, df, food_tabs, mo, on_mp_idx, table):
    ############## Definiere Header für Rezept-Tabellen ###################
    if food_tabs.value in ["Kochen", "Backen", "Sonstiges"]:  
        if not len(table.value) == 0:
            meal_plan_chkbx = mo.ui.checkbox(value = True if df[df['name'] == table.value.name.to_list()[0]].idx.to_list()[0] in on_mp_idx else False, label="Auf Speiseplan", on_change = lambda x: add_meal_to_mp())
            shopping_list_rbttn = mo.ui.run_button(label="Zutaten auf Einkaufsliste")
            show_pdf_chkbx = mo.ui.checkbox(value = False, label="Zeige Rezept pdf")
            _src = df.loc[table.value.index].iloc[0,df.columns.get_loc('pdf')]
            with open(_src, "rb") as _file:
                recipe_pdf_view = mo.pdf(src=_file)
    return (
        meal_plan_chkbx,
        recipe_pdf_view,
        shopping_list_rbttn,
        show_pdf_chkbx,
    )


@app.cell
def _(
    food_tabs,
    meal_plan_chkbx,
    recipe_pdf_view,
    shopping_list_rbttn,
    show_pdf_chkbx,
    table,
):
    ####### Sorge dafür, dass pdf nicht angezeigt wird, wenn die Checkbox nicht aktiviert ist
    if food_tabs.value in ["Kochen", "Backen", "Sonstiges"]:  
        if len(table.value) == 0:
            option_header = []
        elif not show_pdf_chkbx.value:
            option_header = [meal_plan_chkbx, shopping_list_rbttn, show_pdf_chkbx]
        else:
            option_header = [meal_plan_chkbx, shopping_list_rbttn, show_pdf_chkbx, recipe_pdf_view]
    return (option_header,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Steuere Verhalten wenn Zutaten des Rezepts automatisch auf Einkaufliste sollen
        Wenn Zutaten auf Einkaufliste sollen: Nach Knopfdruck werden die Zutaten auf die Einkaufsliste gesetzt. Dabei wird das "pruning" angewendet um gleiche Zutaten zu addieren und nicht doppelt auf die Liste zu setzen.
        """
    )
    return


@app.cell
def _(add_ingredients_to_sl, df, food_tabs, mo, shopping_list_rbttn, table):
    ##### Setze Zutaten auf die Einkaufsliste #################
    ##### Da der Button nur definiert ist, wenn ein Rezept zu sehen ist, muss das vorher überprüft werden ######
    if food_tabs.value in ["Kochen", "Backen", "Sonstiges"]:
        if not len(table.value) == 0:
            if shopping_list_rbttn.value:
                ingredients_str = df.loc[table.value.index].iloc[0,df.columns.get_loc('ingredients')]
                ingredients = [e for e in ingredients_str.split(",")]
                default_sel_str = df.loc[table.value.index].iloc[0,df.columns.get_loc('to_buy_std')]
                default_sel_list = [bool(int(e)) for e in default_sel_str.split(",")]
                default_sel_list = [i for i, x in enumerate(default_sel_list) if x == True]
                ing_table = mo.ui.table(ingredients, initial_selection=default_sel_list, page_size=15, 
                        show_download=False).form(show_clear_button = True, on_change = lambda x: add_ingredients_to_sl(x))
    return (
        default_sel_list,
        default_sel_str,
        ing_table,
        ingredients,
        ingredients_str,
    )


@app.cell
def _(mo):
    mo.md("""### Steuerung des Verhaltens der Einkaufsliste im Hintergrund, wenn Rezepte sichtbar sind.""")
    return


@app.cell
def _(nb_path, pd):
    def load_df_shopping_list():
        return pd.read_csv(nb_path.joinpath('Einkaufsliste.csv'), sep=';', encoding = 'utf8')

    def load_df_shopping_helper_list():
        return pd.read_csv(nb_path.joinpath('Einkaufsliste_Helfer.csv'), sep=';', encoding = 'utf8')

    df_test = load_df_shopping_helper_list()
    return df_test, load_df_shopping_helper_list, load_df_shopping_list


@app.cell
def _(load_df_shopping_list, nb_path, prune_ing_table):
    #### Lade Einkaufsliste und fasse gleiche Artikel zusammen #####
    #### Die neue Einkaufsliste wird dann auch gespeichert #######
    #mo.stop(len(table.value)==0)
    def add_ingredients_to_sl(_ings):
        #print(_ings)
        df_shopping_list = load_df_shopping_list()
        _last_elem = df_shopping_list.shape[0]
        for i in range(len(_ings)):
            _to_add = prune_ing_table(_ings[i])
            df_shopping_list.loc[i+_last_elem] = _to_add
        df_shopping_list = df_shopping_list.groupby(['artikel', 'einheit'])[['menge', 'rang']].aggregate({'menge': 'sum', 'rang': 'median'}).reset_index()
        df_shopping_list = df_shopping_list[['artikel', 'menge', 'einheit', 'rang']]

        ##### save new shopping list ####
        #print('save new shopping list...')
        df_shopping_list.to_csv(nb_path.joinpath('Einkaufsliste.csv'), sep = ';', index = False, encoding = 'utf8')
    return (add_ingredients_to_sl,)


@app.cell
def _(load_df_shopping_helper_list):
    ##### Dieser Block hilft dabei, die Einkaufsliste zu organisieren indem gleiche Artikel zusammengefasst werden #####
    ##### In der Helper-Liste ist der "Rang" gespeichert der angibt, an welcher Stelle die Artikel gekauft werden sollen ######
    import re
    _default_rang_for_shopping_list = 3
    _default_einheit = "x"
    def prune_ing_table(_item):
        _idx = re.search(r"\(\d*[.]*d*", _item)
        if _idx == None:
            _menge = 1
            _einheit = _default_einheit
            _artikel = _item
        else:
            _menge = _item[_idx.start()+1:_idx.end()]
            _einheit = _item[_idx.end():-1]
            _artikel = _item[0:_idx.start()-1]
        if _einheit == "":
            _einheit = _default_einheit
        _df_shopping_list_helper = load_df_shopping_helper_list()
        _helper_entry = _df_shopping_list_helper[_df_shopping_list_helper['artikel'] == _artikel]
        if _helper_entry.empty:
            _rang = _default_rang_for_shopping_list
        else:
            _rang = int(_helper_entry.rang.values[0])
        print([_artikel, _menge, _einheit, _rang])
        return [_artikel, float(_menge), _einheit, _rang]
    return prune_ing_table, re


@app.cell
def _(mo):
    mo.md("""### Funktion, die Rezepte auf den Speiseplan befördert oder entfernt""")
    return


@app.cell
def _(df, nb_path, on_mp_idx, pd, table):
    def add_meal_to_mp():
        _meal_name = table.value.name
        _meal_idx = df[df['name'] == table.value.name.to_list()[0]].idx.to_list()[0]
        if _meal_idx in on_mp_idx:   # Essen war auf Speiseplan - und soll wohl folglich runter davon
            on_mp_idx.remove(_meal_idx)
            df.loc[_meal_idx, 'on_shopping_list'] = False
        else:    # Essen soll auf Speiseplan gesetzt werden
            on_mp_idx.append(_meal_idx)
            df.loc[_meal_idx,'on_shopping_list'] = True
        _on_mp_names = df.loc[on_mp_idx].name.to_list()
        _new_mp = {'name': _on_mp_names, 'idx_rezept': on_mp_idx}
        pd.DataFrame.from_dict(_new_mp).to_csv(nb_path.joinpath('Speiseplan.csv'), index = False, sep = ';', encoding='utf8')
    return (add_meal_to_mp,)


@app.cell
def _(mo):
    mo.md("""# Funktionen und Definitionen für die Einkaufsliste und deren Handling""")
    return


@app.cell
def _(mo):
    mo.md("""### Definiere die Ansicht im Edit-Modus - ACHTUNG: Wird nicht verwendet wegen Problem mit Interaktiven Elementen in Funktionen""")
    return


@app.cell
def _(mo, set_state_sl):
    def show_sl_in_edit_mode(_data_o):
        _data = _data_o.copy(deep=True)
        _article = mo.ui.array(
            [mo.ui.text(label = None, value = n) for n in _data['artikel'] ]
        )
        _numbers = mo.ui.array(
            [mo.ui.number(label = "", start = 0, value=n) for n in _data['menge'] ]
        )
        _unit = mo.ui.array(
            [mo.ui.text(label=None, value = n) for n in _data['einheit']]

        )
        _rang = mo.ui.array(
            [mo.ui.number(label = "", start = 0, value = n) for n in _data['rang'] ]
        )
        _loc_butt = mo.ui.array([
            mo.ui.run_button(label = '-', on_change=lambda v, i=i: set_state_sl(i)) for i in range(_data.shape[0])])
        _data['artikel'] = list(_article)
        _data['remove'] = list(_loc_butt)
        _data['menge'] = list(_numbers)
        _data['einheit'] = list(_unit)
        _data['rang'] = list(_rang)
        return _data
    return (show_sl_in_edit_mode,)


@app.cell
def _(mo):
    mo.md("""### Definiere die Ansicht im Shopping-Modus (nur entfernen möglich - ACHTUNG: Wird nicht mehr verwendet, s.o.""")
    return


@app.cell
def _(mo, set_state_sl):
    def show_sl_in_shopping_mode(_data_o):
        _data = _data_o#.copy(deep=True)  #deep = True
        loc_butt = mo.ui.array([
            mo.ui.run_button(label = '-', on_change=lambda v, i=i: set_state_sl(i)) for i in range(_data.shape[0])])
        _data['remove'] = list(loc_butt)
        return _data
    return (show_sl_in_shopping_mode,)


@app.cell
def _():
    def get_vals(df):
        vals = []
        for i, j in range(df.shape()):
            print(i.value)
    return (get_vals,)


@app.cell
def _(mo):
    mo.md("""### Definiere Funktionen für das Löschen eines Eintrags und für das Hinzufügen""")
    return


@app.cell
def _(
    get_state_old_df,
    get_state_view_changed,
    pd,
    set_state_old_df,
    set_state_sl,
    set_state_view_changed,
):
    def alter_df(how):
        if how != 'None':
            if how >= 0:
                _act_df = get_state_old_df().drop(how)
                set_state_old_df(_act_df.copy(deep = True))
                #set_state_old_df_view(get_state_old_df_view().drop(how))
                set_state_sl(-1)  # change of df finished
                set_state_view_changed(not get_state_view_changed()) # tell view to update

    def add_item_to_sl(_item, _amount, _unit, _rank):
        if _item != '':
            _max_ind = get_state_old_df().index.max()
    #        _act_item = mo.ui.text(value = _item) # item
    #        _act_amount = mo.ui.number(value = _amount) # add amount
    #        _act_unit = mo.ui.text(value = _unit) # add unit
    #        _act_rank = mo.ui.number(value = _rank) # add rank
            _to_add_dict = {'artikel': [_item], 'menge': [_amount], 'einheit': [_unit], 'rang': [_rank], 'idx': [_max_ind+1]}
            _to_add_df = pd.DataFrame.from_dict(_to_add_dict).set_index('idx').copy(deep=True)
            #print(_to_add_df)
            _act_df = pd.concat([get_state_old_df(), _to_add_df])#.reset_index(drop=True)
            set_state_old_df(_act_df.copy(deep = True))
            set_state_view_changed(not get_state_view_changed())
    return add_item_to_sl, alter_df


@app.cell
def _(mo):
    mo.md("""### Definiere Funktion für das Abändern eines einzelnen Eintrags mittels UI-Element im Bearbeiten-Modus""")
    return


@app.cell
def _(get_state_old_df, set_state_old_df):
    def update_sl_df(_col, _row, _v):
        _actval = _v#filtered_df_sl.iat[_row, _col].copy().value
        _tmp_df = get_state_old_df().copy()
        _tmp_df.iat[_row, _col] = _actval
        set_state_old_df(_tmp_df.copy())
    return (update_sl_df,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Definiere das Handling der im Folgenden definierten UI-Elemente.
        Nutze get_state_sl() um Änderungen abzufangen und entsprechend zu handhaben. Standardmäßig ist der Wert auf -1 gesetzt was heißt, dass aktuell keine Änderung vorliegt. Wird ein entfernen-Button gedrückt, ändert sich der Zustand auf den Index des Dataframes, der entfernt werden soll.
        """
    )
    return


@app.cell
def _(alter_df, get_state_sl):
    get_state_sl()

    if get_state_sl() >= 0:
        print(get_state_sl())
        alter_df(get_state_sl())
    return


@app.cell
def _(mo):
    mo.md("""### Definiere UI-Elemente für das Bearbeiten der Einkaufsliste""")
    return


@app.cell
def _(get_state_old_df, mo, nb_path):
    ### Definition für Speichern der neuen Werte
    update_button = mo.ui.run_button(label = 'update files', on_change = lambda x: get_state_old_df().to_csv(nb_path.joinpath('Einkaufsliste.csv'), sep = ';', index = False, encoding = 'utf8'))
    return (update_button,)


@app.cell
def _(food_tabs, mo):
    # Definiere Eingabemaske für Einkaufsliste
    food_tabs.value
    sl_add_item = mo.ui.text(label = 'artikel', value ='')
    sl_add_amount = mo.ui.number(label = 'menge', value = 1)
    sl_add_unit = mo.ui.text(label = 'einheit', value = 'x')
    sl_add_rank = mo.ui.number(label = 'rang', value = 3)
    return sl_add_amount, sl_add_item, sl_add_rank, sl_add_unit


@app.cell
def _(
    add_item_to_sl,
    food_tabs,
    mo,
    sl_add_amount,
    sl_add_item,
    sl_add_rank,
    sl_add_unit,
):
    # Definiere den Submit-Button für die Eingabemaske der Einkaufsliste
    food_tabs.value
    sl_add_submit_button = mo.ui.run_button(label = 'add', on_change = lambda x: add_item_to_sl(sl_add_item.value, sl_add_amount.value, sl_add_unit.value, sl_add_rank.value))
    return (sl_add_submit_button,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
