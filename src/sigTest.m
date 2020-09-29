clear; clc; close all;

sig200 = siggen(200, 8000, 1);
sig300 = siggen(300, 8000, 1);
sound([sig200; sig300], 8000);
