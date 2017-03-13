import java.io.BufferedOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import java.io.*;
import java.nio.*;
import java.util.concurrent.*;

public class FileGen
{
    public static void main(String args[]) throws IOException
    {
        Path path = Paths.get("sim.csv");
        OutputStream out = new BufferedOutputStream(Files.newOutputStream(path, StandardOpenOption.CREATE, StandardOpenOption.APPEND));
        Random generator = new Random();
        String s = "";
        for(int k = 0; k < 5000000; k++)
        {
            s = "";
            for (int i = 0; i < 5; i++) {
                s += (generator.nextDouble()) + ",";
            }
            s += (generator.nextDouble()) + "\n";
            out.write(s.getBytes());
        }

        out.close();
    }
}
