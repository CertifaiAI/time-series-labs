package exercise.stockprediction.utils;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

import javafx.scene.chart.Chart;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Plot {

    private JFrame window;
    private INDArray x;
    private INDArray y;


    public Plot(final INDArray x, final INDArray y, final String title){
        this.window = new JFrame();
        this.window.setTitle(title);
        this.window.setSize(600,400);
        this.window.setLayout(new BorderLayout());
        this.window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        this.x = x;
        this.y = y;

    }

    public void display(){
        XYSeriesCollection dataset = new XYSeriesCollection();
        addSeries(dataset,x,y,"Stock Price Data");
        JFreeChart chart = ChartFactory.createXYLineChart("Stock price data", "Time (day)",
                "Open price ($)", dataset, PlotOrientation.VERTICAL, true, true, false);

        window.add(new ChartPanel(chart), BorderLayout.CENTER);
        window.setVisible(true);
    }

    private void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
        final double[] xd = x.toDoubleVector();
        final double[] yd = y.toDoubleVector();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }

}
