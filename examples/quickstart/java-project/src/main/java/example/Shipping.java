package example;

/** Shipping costs of an online shop: the Java example of Testuary's quickstart. */
public final class Shipping {

    private Shipping() {
    }

    /** The shipping cost of a parcel in euros. */
    public static double cost(double weightKg, boolean express, String country) {
        if (weightKg <= 0) {
            throw new IllegalArgumentException("the weight must be positive");
        }
        double cost;
        if (weightKg <= 2) {
            cost = 4.99;
        } else if (weightKg <= 10) {
            cost = 7.99;
        } else {
            cost = 7.99 + 0.5 * (weightKg - 10);
        }
        if (!"DE".equals(country)) {
            cost += 6.0;
        }
        if (express) {
            cost *= 1.5;
        }
        return Math.round(cost * 100) / 100.0;
    }

    /** Whether an order ships free of charge. */
    public static boolean freeShipping(double orderTotal, boolean member) {
        double threshold = member ? 30.0 : 50.0;
        return orderTotal >= threshold;
    }
}
