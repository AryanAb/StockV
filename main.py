import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from dearpygui import core, simple
import os
import yfinance as yf


def create_model(sender, data):
    print(data)
    print(data['filename'])
    _data = pd.read_csv("dataset/" + data['filename'])
    price = _data[['Close']]

    scalar = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scalar.fit_transform(price['Close'].values.reshape(-1, 1))

    lookback = 20

    x_train, y_train, x_test, y_test = split_data(price, lookback)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    input_dim = data['input_dim']
    hidden_dim = data['hidden_dim']
    num_layers = data['num_layers']
    output_dim = data['output_dim']
    num_epochs = data['num_epochs']

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    #core.log_debug("Training time: {}".format(training_time))

    print(scalar.inverse_transform(y_train_pred.detach().numpy()))
    predict = pd.DataFrame(scalar.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scalar.inverse_transform(y_train_gru.detach().numpy()))

    return predict, original

    # plt.plot(predict.index, predict[0], label="Prediction")
    #global predict
    #predict = _predict

    # plt.plot(original.index, original[0], label="Actual")

    #plt.plot(predict.index, predict[0], label="Prediction")
    # plt.xlabel("Date")
    #plt.ylabel("Stock - USD")
    #plt.title("GRU Model Performance")
    # plt.legend()
    # plt.show()


def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []

    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


selected_companies = []

selected_cap = ""
g_cap_interval = ""
g_draw_cap = True

colors = [[50, 200, 50, 100], [50, 50, 200, 100], [200, 50, 50, 100]]

model_company = ""
g_input_dim = 1
g_hidden_dim = 32
g_num_layers = 2
g_output_dim = 1
g_num_epochs = 10
g_fast_mode = False


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


def show_cap_level(sender, data):
    global g_draw_cap
    g_draw_cap = core.get_value("Show Cap Levels")
    cap_callback(None, g_cap_interval)


def cap_callback(sender, data):
    global selected_cap
    selected_cap = data
    core.clear_plot("Cap")

    global g_cap_interval
    g_cap_interval = data

    core.set_value("Cap Interval##plot", "Cap Interval: " + data.capitalize())

    for i in range(0, len(selected_companies)):
        company = selected_companies[i]
        stocks = pd.read_csv("dataset/" + company, parse_dates=['Date'])
        stocks['Date'] = pd.to_datetime(stocks['Date'], unit='D', errors='coerce')

        if data == "daily":
            data_x = list(range(1, len(stocks.index) + 1))
            data_y = (stocks['High'] * stocks['Volume']).tolist()
            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

        elif data == "monthly":
            monthly_stocks = stocks.groupby(pd.Grouper(key="Date", freq='1M')).mean()
            data_x = list(range(1, len(monthly_stocks.index) + 1))
            data_y = (monthly_stocks['High'] * monthly_stocks['Volume']).tolist()
            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

        elif data == "quarterly":
            quarterly_stocks = stocks.groupby(pd.Grouper(key="Date", freq='3M')).mean()
            data_x = list(range(1, len(quarterly_stocks.index) + 1))
            data_y = (quarterly_stocks['High'] * quarterly_stocks['Volume']).tolist()
            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

        elif data == "yearly":
            yearly_stocks = stocks.groupby(pd.Grouper(key="Date", freq='1Y')).mean()
            data_x = list(range(1, len(yearly_stocks.index) + 1))
            data_y = (yearly_stocks['High'] * yearly_stocks['Volume']).tolist()
            core.add_line_series("Cap", company.split('_')[0], data_x, data_y, color=colors[i % len(colors)])

        if g_draw_cap:
            if data == "daily":
                core.add_line_series("Cap", "large-cap", [1, 3020], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
                core.add_line_series("Cap", "mid-cap", [1, 3020], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
                core.add_line_series("Cap", "small-cap", [1, 3020], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])
            elif data == "monthly":
                core.add_line_series("Cap", "large-cap", [1, 144], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
                core.add_line_series("Cap", "mid-cap", [1, 144], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
                core.add_line_series("Cap", "small-cap", [1, 144], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])
            elif data == "quarterly":
                core.add_line_series("Cap", "large-cap", [1, 49], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
                core.add_line_series("Cap", "mid-cap", [1, 49], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
                core.add_line_series("Cap", "small-cap", [1, 49], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])
            elif data == "yearly":
                core.add_line_series("Cap", "large-cap", [1, 12], [10000000000, 10000000000], weight=3, color=[255, 50, 50, 100])
                core.add_line_series("Cap", "mid-cap", [1, 12], [2000000000, 2000000000], weight=3, color=[200, 50, 50, 100])
                core.add_line_series("Cap", "small-cap", [1, 12], [300000000, 300000000], weight=3, color=[150, 50, 50, 100])


def clear(sender, data):
    plot_callback(None, data)
    cap_callback(None, data)


def set_input_dim(sender, data):
    global g_input_dim
    g_input_dim = core.get_value("Input Dim")


def set_hidden_dim(sender, data):
    global g_hidden_dim
    g_hidden_dim = core.get_value("Hidden Dim")


def set_num_layers(sender, data):
    global g_num_layers
    g_num_layers = core.get_value("Num Layers")


def set_output_dim(sender, data):
    global g_output_dim
    g_output_dim = core.get_value("Output Dim")


def set_num_epochs(sender, data):
    global g_num_epochs
    g_num_epochs = core.get_value("Num Epochs")


def set_fast_mode(sender, data):
    global g_fast_mode
    g_fast_mode = core.get_value("Fast Mode")


def update_model_window(sender, data):
    global model_company
    model_company = data
    core.set_value("company_id", "Company: " + data.split('_')[0])
    core.configure_item("Create Model", enabled=True)


def create_btn(sender, data):
    core.set_value("company_id", "Creating {} Model ...".format(model_company.split('_')[0]))
    dict = {
        "filename": model_company,
        "input_dim": g_input_dim,
        "hidden_dim": g_hidden_dim,
        "num_layers": g_num_layers,
        "output_dim": g_output_dim,
        "num_epochs": g_num_epochs
    }
    if not g_fast_mode:
        core.run_async_function(create_model, dict, return_handler=create)
    else:
        create(None, create_model(None, dict))


def create(sender, data):
    core.set_value("company_id", "{} Model Created".format(model_company.split('_')[0]))
    core.clear_plot("Pred")
    predict, original = data
    core.add_line_series("Pred", "Prediction", predict.index.tolist(), predict[0].tolist(), color=[255, 50, 50, 100])


with simple.window("Stock", width=650, height=300, x_pos=10, y_pos=110, no_close=True):
    core.add_plot("Plot", height=-1)

with simple.window("Market Cap", width=650, height=375, x_pos=10, y_pos=415, no_close=True):
    core.add_text(name="Cap Interval##plot", default_value="Cap Interval: None")
    core.add_checkbox("Show Cap Levels", default_value=True, callback=show_cap_level)
    core.add_plot("Cap", height=-1)


with simple.window("Select Company", height=90, x_pos=10, y_pos=5, no_close=True):
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

with simple.window("Model", height=300, width=300, x_pos=675, y_pos=110, no_close=True):
    with simple.menu("Select Company##model"):
        for file in os.listdir("dataset"):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                core.add_menu_item(filename.split('_')[0] + "##model", callback_data=filename, callback=update_model_window)

    core.add_text(name="company_id", default_value="Company: None")
    core.add_drag_int("Input Dim", default_value=1, callback=set_input_dim)
    core.add_drag_int("Hidden Dim", default_value=32, callback=set_hidden_dim)
    core.add_drag_int("Num Layers", default_value=2, callback=set_num_layers)
    core.add_drag_int("Output Dim", default_value=1, callback=set_output_dim)
    core.add_drag_int("Num Epochs", default_value=10, callback=set_num_epochs)
    core.add_checkbox("Fast Mode", default_value=False, callback=set_fast_mode)
    core.add_button("Create Model", enabled=False, callback=create_btn)

with simple.window("Prediction", width=650, height=375, x_pos=675, y_pos=415, no_close=True):
    core.add_plot("Pred", height=-1)


def update_table(sender, data):
    ticker = yf.Ticker(data)
    calendar = ticker.calendar
    core.set_table_item("Earnings", 0, 1, calendar.iloc[0, 0].ctime())
    core.set_table_item("Earnings", 1, 1, repr(calendar.iloc[1, 0]))
    core.set_table_item("Earnings", 2, 1, repr(calendar.iloc[2, 0]))
    core.set_table_item("Earnings", 3, 1, repr(calendar.iloc[3, 0]))
    core.set_table_item("Earnings", 4, 1, repr(calendar.iloc[4, 0]))
    core.set_table_item("Earnings", 5, 1, repr(calendar.iloc[5, 0]))
    core.set_table_item("Earnings", 6, 1, repr(calendar.iloc[6, 0]))


with simple.window("Tables", height=300, width=500, x_pos=980, y_pos=110, no_close=True):
    with simple.menu("Select Company##table"):
        for file in os.listdir("dataset"):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                core.add_menu_item(filename.split('_')[0] + "##table", callback_data=filename.split('_')[0], callback=update_table)
    core.add_table("Earnings", ["Specs", "Value"])
    core.add_row("Earnings", ["Earnings Date", ""])
    core.add_row("Earnings", ["Earnings Average", ""])
    core.add_row("Earnings", ["Earnings Low", ""])
    core.add_row("Earnings", ["Earnings High", ""])
    core.add_row("Earnings", ["Revenue Average", ""])
    core.add_row("Earnings", ["Revenue Low", ""])
    core.add_row("Earnings", ["Revenue High", ""])

# core.show_logger()
# simple.show_documentation()

core.start_dearpygui()
