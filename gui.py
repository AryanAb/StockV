from dearpygui import core, simple
import pandas as pd
import os
import model

selected_companies = []
selected_cap = ""
colors = [[50, 200, 50, 100], [50, 50, 200, 100], [200, 50, 50, 100]]

model_company = ""


def plot_callback(sender, data):

    if data == "clear":
        selected_companies.clear()
    elif data != None:
        if data not in selected_companies:
            selected_companies.append(data)
        else:
            selected_companies.remove(data)

    core.clear_plot("Plot")
    cap_callback(None, selected_cap)

    for i in range(0, len(selected_companies)):
        company = selected_companies[i]
        stocks = pd.read_csv("dataset/" + company)

        data_x = list(range(1, len(stocks.index) + 1))
        data_y = stocks['High'].tolist()
        core.add_line_series("Plot", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])


def cap_callback(sender, data):
    global selected_cap
    selected_cap = data
    core.clear_plot("Cap")

    for i in range(0, len(selected_companies)):
        company = selected_companies[i]
        stocks = pd.read_csv("dataset/" + company, parse_dates=['Date'])
        stocks['Date'] = pd.to_datetime(stocks['Date'], unit='D', errors='coerce')

        if data == "daily":
            data_x = list(range(1, len(stocks.index) + 1))
            data_y = (stocks['High'] * stocks['Volume']).tolist()
            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

            core.add_line_series("Cap", "large-cap", [data_x[0], data_x[-1]], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
            core.add_line_series("Cap", "mid-cap", [data_x[0], data_x[-1]], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
            core.add_line_series("Cap", "small-cap", [data_x[0], data_x[-1]], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])
        elif data == "monthly":
            monthly_stocks = stocks.groupby(pd.Grouper(key="Date", freq='1M')).mean()
            data_x = list(range(1, len(monthly_stocks.index) + 1))
            data_y = (monthly_stocks['High'] * monthly_stocks['Volume']).tolist()

            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

            core.add_line_series("Cap", "large-cap", [data_x[0], data_x[-1]], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
            core.add_line_series("Cap", "mid-cap", [data_x[0], data_x[-1]], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
            core.add_line_series("Cap", "small-cap", [data_x[0], data_x[-1]], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])
        elif data == "quarterly":
            quarterly_stocks = stocks.groupby(pd.Grouper(key="Date", freq='3M')).mean()
            data_x = list(range(1, len(quarterly_stocks.index) + 1))
            data_y = (quarterly_stocks['High'] * quarterly_stocks['Volume']).tolist()

            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

            core.add_line_series("Cap", "large-cap", [data_x[0], data_x[-1]], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
            core.add_line_series("Cap", "mid-cap", [data_x[0], data_x[-1]], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
            core.add_line_series("Cap", "small-cap", [data_x[0], data_x[-1]], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])
        elif data == "yearly":
            yearly_stocks = stocks.groupby(pd.Grouper(key="Date", freq='1Y')).mean()
            data_x = list(range(1, len(yearly_stocks.index) + 1))
            data_y = (yearly_stocks['High'] * yearly_stocks['Volume']).tolist()

            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

            core.add_line_series("Cap", "large-cap", [data_x[0], data_x[-1]], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
            core.add_line_series("Cap", "mid-cap", [data_x[0], data_x[-1]], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
            core.add_line_series("Cap", "small-cap", [data_x[0], data_x[-1]], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])


def clear(sender, data):
    plot_callback(None, data)
    cap_callback(None, data)


def update_model_window(sender, data):
    global model_company
    model_company = data
    core.set_value("company_id", "Company: " + data.split('_')[0])


def create_model(sender, data):
    predict, original = model.create_model(model_company)
    print(predict.index)
    print(predict[0])
    print(type(predict.index))
    print(type(predict[0]))
    core.add_line_series("Pred", "Prediction", predict.index.tolist(), predict[0].tolist(), color=[255, 50, 50, 100])


with simple.window("Stock", width=650, height=300, x_pos=10, y_pos=110):
    core.add_plot("Plot", height=-1)

with simple.window("Market Cap", width=650, height=300, x_pos=10, y_pos=415):
    core.add_plot("Cap", height=-1)


with simple.window("Select Company", height=90, x_pos=10, y_pos=5):
    with simple.menu("Companies##plot"):
        for file in os.listdir("dataset"):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                core.add_menu_item(filename.split('_')[0] + "##plot", callback_data=filename, callback=plot_callback)

    with simple.menu("Cap Interval"):
        core.add_menu_item("Daily", callback_data="daily", callback=cap_callback)
        core.add_menu_item("Monthly", callback_data="monthly", callback=cap_callback)
        core.add_menu_item("Quarterly", callback_data="quarterly", callback=cap_callback)
        core.add_menu_item("Yearly", callback_data="yearly", callback=cap_callback)

    core.add_button("Reset", callback=plot_callback)
    core.add_same_line()
    core.add_button("Clear", callback_data="clear", callback=clear)

with simple.window("Model", height=90, x_pos=675, y_pos=5):
    with simple.menu("Select Company##model"):
        for file in os.listdir("dataset"):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                core.add_menu_item(filename.split('_')[0] + "##model", callback_data=filename, callback=update_model_window)

    core.add_text(name="company_id", default_value="Company: None")

    core.add_button("Create Model", callback=create_model)

with simple.window("Prediction", width=650, height=300, x_pos=675, y_pos=110):
    core.add_plot("Pred", height=-1)

# core.show_logger()

core.start_dearpygui()
