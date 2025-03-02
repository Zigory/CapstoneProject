import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")

class FinancialAdvisorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPY Financial Advisor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Initialize data
        self.data = None

        # Create main frames
        self.create_header_frame()
        self.create_main_content()
        self.create_footer()

    def create_header_frame(self):
        header_frame = tk.Frame(self.root, bg="white", height=70)
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(header_frame, text="SPY Financial Advisor",
                               font=("Arial", 22, "bold"), bg="blue", fg="white")
        title_label.pack(pady=15)

    def create_main_content(self):
        # main content area
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # left panel (controls)
        left_panel = tk.Frame(main_frame, bg="#ecf0f1", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        #add control widgets
        control_label = tk.Label(left_panel, text="Controls", font=("Arial", 14, "bold"),
                                 bg="#ecf0f1", fg="#2c3e50")
        control_label.pack(pady=10)

        # File selection
        file_frame = tk.Frame(left_panel, bg="#ecf0f1")
        file_frame.pack(fill=tk.X, pady=5)

        file_button = ttk.Button(file_frame, text="Load CSV File", command=self.load_data)
        file_button.pack(fill=tk.X, pady=5)

        # SMA Period selection
        sma_frame = tk.Frame(left_panel, bg="#ecf0f1")
        sma_frame.pack(fill=tk.X, pady=5)

        sma_label = tk.Label(sma_frame, text="SMA Period:", bg="#ecf0f1")
        sma_label.pack(side=tk.LEFT, pady=5)

        self.sma_period = tk.StringVar(value="200")
        sma_entry = ttk.Entry(sma_frame, textvariable=self.sma_period, width=10)
        sma_entry.pack(side=tk.RIGHT, pady=5)

        # Future days for prediction
        future_frame = tk.Frame(left_panel, bg="#ecf0f1")
        future_frame.pack(fill=tk.X, pady=5)

        future_label = tk.Label(future_frame, text="Future Days:", bg="#ecf0f1")
        future_label.pack(side=tk.LEFT, pady=5)

        #Initialize future days to 10
        self.future_days = tk.StringVar(value="10")
        future_entry = ttk.Entry(future_frame, textvariable=self.future_days, width=10)
        future_entry.pack(side=tk.RIGHT, pady=5)

        # Analysis button
        analyze_button = ttk.Button(left_panel, text="Run Analysis", command=self.run_analysis)
        analyze_button.pack(fill=tk.X, pady=10)

        # Portfolio simulation
        portfolio_frame = tk.Frame(left_panel, bg="#ecf0f1")
        portfolio_frame.pack(fill=tk.X, pady=5)

        portfolio_label = tk.Label(portfolio_frame, text="Initial Investment ($):", bg="#ecf0f1")
        portfolio_label.pack(side=tk.LEFT, pady=5)

        self.initial_investment = tk.StringVar(value="10000")
        portfolio_entry = ttk.Entry(portfolio_frame, textvariable=self.initial_investment, width=10)
        portfolio_entry.pack(side=tk.RIGHT, pady=5)

        # Right panel (results and charts)
        self.right_panel = tk.Frame(main_frame, bg="white")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.overview_tab = ttk.Frame(self.notebook)
        self.price_chart_tab = ttk.Frame(self.notebook)
        self.strategy_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.recommendation_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.price_chart_tab, text="Price Charts")
        self.notebook.add(self.strategy_tab, text="Strategy Comparison")
        self.notebook.add(self.model_tab, text="ML Model")
        self.notebook.add(self.recommendation_tab, text="Recommendations")

        # Initialize overview tab with the welcome message
        welcome_frame = tk.Frame(self.overview_tab, bg="white")
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        welcome_label = tk.Label(welcome_frame,
                                 text="Welcome to Zigory Financial's S&P 500 Financial Advisor\n\nLoad a CSV file and run the analysis to get started.",
                                 font=("Arial", 14), bg="white", justify=tk.CENTER)
        welcome_label.pack(pady=100)

    def create_footer(self):
        footer_frame = tk.Frame(self.root, bg="#2c3e50", height=30)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)

        footer_label = tk.Label(footer_frame, text="S&P 500 Financial Advisor - For Educational Purposes Only",
                                font=("Arial", 8), bg="#2c3e50", fg="white")
        footer_label.pack(pady=5)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try: #data = pd.read_csv('spy.csv', index_col=0, parse_dates = True)
                self.data = pd.read_csv(file_path)

                if 'Date' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Date'])
                    # Set Date as index
                    self.data.set_index('Date', inplace=True)


                messagebox.showinfo("Success", f"Loaded data with {len(self.data)} records")

                # Display data summary in overview tab
                self.update_overview_tab()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def update_overview_tab(self):
        # Clear the tab
        for widget in self.overview_tab.winfo_children():
            widget.destroy()

        if self.data is None:
            return

        # create frames for different sections
        header_frame = tk.Frame(self.overview_tab, bg="white")
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        summary_frame = tk.Frame(self.overview_tab, bg="white")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        #header with data info
        header_label = tk.Label(header_frame, text=f"SPY Data Overview", font=("Arial", 16, "bold"), bg="white")
        header_label.pack(anchor=tk.W, pady=5)

        date_range = f"Date Range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}"
        date_label = tk.Label(header_frame, text=date_range, font=("Arial", 12), bg="white")
        date_label.pack(anchor=tk.W, pady=2)

        # Data summary in scrolled text widget
        summary_label = tk.Label(summary_frame, text="Data Summary:", font=("Arial", 12, "bold"), bg="white")
        summary_label.pack(anchor=tk.W, pady=5)

        summary_text = scrolledtext.ScrolledText(summary_frame, width=80, height=20)
        summary_text.pack(fill=tk.BOTH, expand=True)

        # Insert data description
        summary_stats = self.data.describe().to_string()
        summary_text.insert(tk.END, f"Dataset Shape: {self.data.shape}\n\n")
        summary_text.insert(tk.END, f"First 5 rows:\n{self.data.head().to_string()}\n\n")
        summary_text.insert(tk.END, f"Statistical Summary:\n{summary_stats}\n")
        summary_text.config(state=tk.DISABLED)

    def run_analysis(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        try:
            sma_period = int(self.sma_period.get())
            future_days = int(self.future_days.get())
            initial_investment = float(self.initial_investment.get())

            # run analysis
            self.analyze_data(sma_period, future_days, initial_investment)

            # update all tabs with new results
            self.update_price_charts_tab()
            self.update_strategy_tab()
            self.update_model_tab()
            self.update_recommendation_tab()

            # Switch to the price charts tab
            self.notebook.select(self.price_chart_tab)

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def analyze_data(self, sma_period, future_days, initial_investment):
        # run the analysis on the loaded data
        # make a copy of the data to avoid modifying the original
        self.processed_data = self.data.copy()

        # Calculate the 200D SMA
        self.processed_data['SMA'] = self.processed_data['Close'].rolling(window=sma_period).mean()
        self.processed_data['Above_SMA'] = self.processed_data['Close'] > self.processed_data['SMA']
        self.processed_data['Above_SMA_7D'] = self.processed_data['Above_SMA'].rolling(
            window=7).sum() == 7  # True if all 7 days above
        self.processed_data['Below_SMA_7D'] = self.processed_data['Above_SMA'].rolling(
            window=7).sum() == 0  # True if all 7 days below

        # generate  trade signals
        self.processed_data['Signal'] = 1
        self.processed_data.loc[(self.processed_data['Close'] > self.processed_data['SMA']) &
                                (self.processed_data['Above_SMA_7D']), 'Signal'] = 1
        self.processed_data.loc[(self.processed_data['Close'] < self.processed_data['SMA']) &
                                (self.processed_data['Below_SMA_7D']), 'Signal'] = -1

        # create trade signals for crossover events only
        self.processed_data['Prev_Signal'] = self.processed_data['Signal'].shift(1)
        self.processed_data['Trade'] = 0
        self.processed_data.loc[(self.processed_data['Signal'] == 1) &
                                (self.processed_data['Prev_Signal'] == -1), 'Trade'] = 1  # Buy signal
        self.processed_data.loc[(self.processed_data['Signal'] == -1) &
                                (self.processed_data['Prev_Signal'] == 1), 'Trade'] = -1  # Sell signal

        # calculate future returns
        self.processed_data['Future_Return'] = self.processed_data['Close'].shift(-future_days) / self.processed_data[
            'Close'] - 1

        # label outcomes
        self.processed_data['Outcome'] = self.processed_data.apply(self.label_outcome, axis=1)

        # Features
        self.processed_data['Price_SMA_Diff'] = self.processed_data['Close'] - self.processed_data['SMA']

        # daily returns
        self.processed_data['Daily_Returns'] = self.processed_data['Close'].pct_change()

        # buy and hold strategy
        self.processed_data['Buy_Hold'] = (1 + self.processed_data['Daily_Returns']).cumprod()
        self.processed_data['Buy_Hold_Portfolio'] = initial_investment * self.processed_data['Buy_Hold']

        # SMA strategy returns
        self.processed_data['Position'] = self.processed_data['Signal'].shift(1).fillna(0).apply(
            lambda x: 1 if x == 1 else 0)
        self.processed_data['Strategy_Return'] = self.processed_data['Daily_Returns'] * self.processed_data['Position']
        self.processed_data['Strategy_Cumulative_Return'] = (1 + self.processed_data['Strategy_Return']).cumprod()
        self.processed_data['Strategy_Portfolio'] = initial_investment * self.processed_data[
            'Strategy_Cumulative_Return']

        self.processed_data['2X_Returns'] = self.processed_data['Daily_Returns'] * 2
        self.processed_data['2X_Buy_Hold'] = (1 + self.processed_data['2X_Returns']).cumprod()
        self.processed_data['2X_Buy_Hold_Portfolio'] = initial_investment * self.processed_data['2X_Buy_Hold']

        # Leveraged ETF with SMA strategy
        self.processed_data['2X_Strategy_Return'] = self.processed_data['2X_Returns'] * self.processed_data['Position']
        self.processed_data['2X_Strategy_Cumulative_Return'] = (1 + self.processed_data['2X_Strategy_Return']).cumprod()
        self.processed_data['2X_Strategy_Portfolio'] = initial_investment * self.processed_data[
            '2X_Strategy_Cumulative_Return']

        # Calculate drawdowns
        self.processed_data['Buy_Hold_Running_Max'] = self.processed_data['Buy_Hold'].cummax()
        self.processed_data['Buy_Hold_Drawdown'] = self.processed_data['Buy_Hold'] / self.processed_data[
            'Buy_Hold_Running_Max'] - 1

        self.processed_data['Strategy_Running_Max'] = self.processed_data['Strategy_Cumulative_Return'].cummax()
        self.processed_data['Strategy_Drawdown'] = self.processed_data['Strategy_Cumulative_Return'] / \
                                                   self.processed_data['Strategy_Running_Max'] - 1

        self.processed_data['2X_Buy_Hold_Running_Max'] = self.processed_data['2X_Buy_Hold'].cummax()
        self.processed_data['2X_Buy_Hold_Drawdown'] = self.processed_data['2X_Buy_Hold'] / self.processed_data[
            '2X_Buy_Hold_Running_Max'] - 1

        self.processed_data['2X_Strategy_Running_Max'] = self.processed_data['2X_Strategy_Cumulative_Return'].cummax()
        self.processed_data['2X_Strategy_Drawdown'] = self.processed_data['2X_Strategy_Cumulative_Return'] / \
                                                      self.processed_data['2X_Strategy_Running_Max'] - 1


        # Generate recommendations
        self.processed_data['Recommendation'] = np.where(self.processed_data['Close'] > self.processed_data['SMA'],
                                                         'Buy/Hold', 'Sell')

        self.leveraged_data = self.processed_data[self.processed_data.index >= '2006-01-01'].copy()

        # train ML model
        self.train_model()

    def label_outcome(self, row):
        #Label trade outcomes
        if row['Trade'] == 1:
            return 1 if row['Future_Return'] > 0 else 0
        elif row['Trade'] == -1:
            return 1 if row['Future_Return'] < 0 else 0
        else:
            return np.nan

    def train_model(self):
        #Train a machine learning model on the processed data
        # Drop rows without outcomes
        model_data = self.processed_data.dropna(subset=['Outcome'])

        # Select features
        features = ['Close', 'SMA', 'Price_SMA_Diff']
        X = model_data[features]
        y = model_data['Outcome'].astype(int)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train the model
        self.clf = RandomForestClassifier(n_estimators=500, random_state=42)
        self.clf.fit(X_train, y_train)

        # Make predictions
        self.y_pred = self.clf.predict(X_test)
        self.y_test = y_test

        # Store feature importances
        self.feature_importances = self.clf.feature_importances_
        self.feature_names = features

    def update_price_charts_tab(self):
        #Update the price charts tab with visualizations
        # Clear the tab
        for widget in self.price_chart_tab.winfo_children():
            widget.destroy()

        # Create a frame for the chart
        chart_frame = tk.Frame(self.price_chart_tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)
        fig.subplots_adjust(hspace=0.5)

        # Plot data
        axes[0].plot(self.processed_data['Close'], label='Close Price', color='blue')
        axes[0].plot(self.processed_data['SMA'], label=f'SMA ({self.sma_period.get()}-day)', color='red')
        axes[0].set_title('SPY Close Price and SMA')
        axes[0].legend()

        axes[1].plot(self.processed_data['Open'], label='Open', color='green')
        axes[1].set_title('SPY Open Price')

        axes[2].plot(self.processed_data['High'], label='High', color='red')
        axes[2].set_title('SPY High Price')

        axes[3].plot(self.processed_data['Low'], label='Low', color='purple')
        axes[3].set_title('SPY Low Price')

        axes[4].plot(self.processed_data['Volume'], label='Volume', color='orange')
        axes[4].set_title('SPY Volume')

        axes[5].plot(self.processed_data['Daily_Returns'], label='Daily Returns', color='black')
        axes[5].set_title('SPY Daily Returns')

        # Add a canvas to display the figure
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def update_strategy_tab(self):
        # Update the strategy comparison tab
        # Clear the tab
        for widget in self.strategy_tab.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.strategy_tab)
        scrollbar = tk.Scrollbar(self.strategy_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create frames within the scrollable frame
        returns_frame = tk.Frame(scrollable_frame)
        returns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        portfolio_frame = tk.Frame(scrollable_frame)
        portfolio_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 5))

        drawdown_frame = tk.Frame(scrollable_frame)
        drawdown_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        # Smaller figure sizes
        fig1, ax1 = plt.subplots(figsize=(10, 3))

        # Create returns comparison chart
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(self.processed_data.index, self.processed_data['Buy_Hold'], label='Buy & Hold')
        ax1.plot(self.processed_data.index, self.processed_data['Strategy_Cumulative_Return'],
                 label=f'{self.sma_period.get()}-Day SMA Strategy')
        ax1.plot(self.processed_data.index, self.processed_data['2X_Buy_Hold'],
                 label='2X Leveraged Buy & Hold')
        ax1.plot(self.processed_data.index, self.processed_data['2X_Strategy_Cumulative_Return'],
                 label=f'2X Leveraged {self.sma_period.get()}-Day SMA Strategy')
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()

        canvas1 = FigureCanvasTkAgg(fig1, master=returns_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create portfolio value comparison chart
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(self.processed_data.index, self.processed_data['Buy_Hold_Portfolio'], label='Buy & Hold Portfolio')
        ax2.plot(self.processed_data.index, self.processed_data['Strategy_Portfolio'],
                 label=f'{self.sma_period.get()}-Day SMA Strategy Portfolio')
        ax2.plot(self.processed_data.index, self.processed_data['2X_Buy_Hold_Portfolio'],
                 label='2X Leveraged Buy & Hold Portfolio')
        ax2.plot(self.processed_data.index, self.processed_data['2X_Strategy_Portfolio'],
                 label=f'2X Leveraged {self.sma_period.get()}-Day SMA Strategy Portfolio')
        ax2.set_title(f'Portfolio Value (${self.initial_investment.get()} Initial Investment)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()

        canvas2 = FigureCanvasTkAgg(fig2, master=portfolio_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create drawdown comparison chart
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(self.processed_data.index, self.processed_data['Buy_Hold_Drawdown'], label='Buy & Hold Drawdown',
                 color='blue')
        ax3.plot(self.processed_data.index, self.processed_data['Strategy_Drawdown'],
                 label=f'{self.sma_period.get()}-Day SMA Strategy Drawdown', color='red')
        ax3.set_title('Drawdown Comparison')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        ax3.legend()

        canvas3 = FigureCanvasTkAgg(fig3, master=drawdown_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_model_tab(self):
        #Update the ML model tab with results
        # Clear the tab
        for widget in self.model_tab.winfo_children():
            widget.destroy()

        # Create frames for metrics and charts
        metrics_frame = tk.Frame(self.model_tab)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)

        chart_frame = tk.Frame(self.model_tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Add model metrics
        metrics_label = tk.Label(metrics_frame, text="Machine Learning Model Results", font=("Arial", 14, "bold"))
        metrics_label.pack(anchor=tk.W, pady=5)

        metrics_text = scrolledtext.ScrolledText(metrics_frame, width=80, height=15)
        metrics_text.pack(fill=tk.X)

        explanation_text = (
            "Explanation:\n\n"
            "The confusion matrix shows the counts of correct and incorrect predictions for each class.\n"
            "Confusion Matrix layout:\n"
            " [True Positives for Sell Signals] [False Positives for Sell Signals]\n"
            " [False Negatives for Buy Signals] [True Negatives for Buy Signals]\n"
            "The classification report provides precision, recall, and F1-score. For example, high precision "
            "means that when the model predicts a class, it is usually correct; high recall indicates that the "
            "model correctly identifies most of the actual instances of that class; and the F1-score balances both metrics. "
            "These metrics help assess where the model might be improved, such as adjusting thresholds or addressing class imbalance."
        )
        explanation_label = tk.Label(metrics_frame, text=explanation_text,
                                     font=("Arial", 11), wraplength=500, justify=tk.LEFT)
        explanation_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)


        # Calculate and display confusion matrix and classification report
        cm = confusion_matrix(self.y_test, self.y_pred)
        cr = classification_report(self.y_test, self.y_pred)

        metrics_text.insert(tk.END, "Model: Random Forest Classifier\n\n")
        metrics_text.insert(tk.END, f"Confusion Matrix:\n{cm}\n\n")
        metrics_text.insert(tk.END, f"Classification Report:\n{cr}\n")
        metrics_text.config(state=tk.DISABLED)

        # Create feature importance chart
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(self.feature_importances)

        ax.barh(range(len(indices)), self.feature_importances[indices], color='blue', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_title('Feature Importances')
        ax.set_xlabel('Relative Importance')

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_recommendation_tab(self):
        #Update the recommendations tab
        # Clear the tab
        for widget in self.recommendation_tab.winfo_children():
            widget.destroy()

        # Get the latest data points
        latest_data = self.processed_data.iloc[-5:].copy()

        # Create header frame
        header_frame = tk.Frame(self.recommendation_tab, bg="white")
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Latest recommendation
        latest_rec = latest_data.iloc[-1]['Recommendation']
        rec_color = "green" if latest_rec == "Buy/Hold" else "red"

        rec_label = tk.Label(header_frame, text="CURRENT RECOMMENDATION:", font=("Arial", 16, "bold"), bg="white")
        rec_label.pack(pady=(5, 5))

        big_rec_label = tk.Label(header_frame, text=latest_rec, font=("Arial", 24, "bold"), fg="white", bg=rec_color)
        big_rec_label.pack(pady=5, ipadx=10, ipady=5)

        # Latest prices and indicators
        latest_price = latest_data.iloc[-1]['Close']
        latest_sma = latest_data.iloc[-1]['SMA']

        price_frame = tk.Frame(header_frame, bg="white")
        price_frame.pack(fill=tk.X, pady=10)

        price_label = tk.Label(price_frame, text=f"Latest Price: ${latest_price:.2f}", font=("Arial", 14), bg="white")
        price_label.pack(side=tk.LEFT, padx=20)

        sma_label = tk.Label(price_frame, text=f"{self.sma_period.get()}-day SMA: ${latest_sma:.2f}",
                             font=("Arial", 14), bg="white")
        sma_label.pack(side=tk.RIGHT, padx=20)

        # Recent trend section
        trend_frame = tk.Frame(self.recommendation_tab, bg="#f8f9fa", height = 400)
        trend_frame.pack(fill=tk.X, padx=20, pady=10)
        trend_frame.pack_propagate(False)


        trend_label = tk.Label(trend_frame, text="Recent Price Trend", font=("Arial", 14, "bold"), bg="#f8f9fa")
        trend_label.pack(anchor=tk.W, pady=10)

        # Plot recent price trend (dimensions are width, height in inches)
        fig, ax = plt.subplots(figsize=(8, 5))

        # Get the last 800 data points for the chart
        recent_data = self.processed_data.iloc[-800:]

        ax.plot(recent_data.index, recent_data['Close'], label='Close Price', linewidth=2)
        ax.plot(recent_data.index, recent_data['SMA'], label=f'{self.sma_period.get()}-day SMA', linewidth=2,
                color='red')

        # Add buy/sell signals
        buy_signals = recent_data[recent_data['Trade'] == 1]
        sell_signals = recent_data[recent_data['Trade'] == -1]

        ax.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=75, label='Buy Signal')
        ax.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=75, label='Sell Signal')

        ax.set_title('Recent Price Trend with Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=trend_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # summary section
        summary_frame = tk.Frame(self.recommendation_tab, bg="white")
        summary_frame.pack(fill=tk.X, padx=20, pady=10)

        # add strategy performance metrics
        strategy_return = self.processed_data['Strategy_Cumulative_Return'].iloc[-1] * 100 - 100
        buy_hold_return = self.processed_data['Buy_Hold'].iloc[-1] * 100 - 100

        max_strategy_dd = self.processed_data['Strategy_Drawdown'].min() * 100
        max_buy_hold_dd = self.processed_data['Buy_Hold_Drawdown'].min() * 100

        lev_strategy_return = self.processed_data['2X_Strategy_Cumulative_Return'].iloc[-1] * 100 - 100
        lev_buy_hold_return = self.processed_data['2X_Buy_Hold'].iloc[-1] * 100 - 100

        summary_text = f"""
        Strategy Summary:
        SMA Strategy Total Return: {strategy_return:.2f}%
        Buy & Hold Total Return: {buy_hold_return:.2f}%
        2X Leveraged SMA Strategy Total Return: {lev_strategy_return:.2f}%
        2X Leveraged Buy & Hold Total Return: {lev_buy_hold_return:.2f}%

        SMA Strategy Max Drawdown: {max_strategy_dd:.2f}%
        Buy & Hold Max Drawdown: {max_buy_hold_dd:.2f}%

        Note: 2X leveraged ETFs may have higher volatility and tracking errors over time.
        Disclaimer: This is for educational purposes only. Past performance is not indicative of future results.
        """



        summary_box = scrolledtext.ScrolledText(summary_frame, width=80, height=10, font=("Arial", 11))
        summary_box.pack(fill=tk.X, pady=10)
        summary_box.insert(tk.END, summary_text)
        summary_box.config(state=tk.DISABLED)

# run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialAdvisorApp(root)
    root.mainloop()