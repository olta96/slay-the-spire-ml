package RecommenderSystem;

import java.util.ArrayList;

public class Deck {

    private final ArrayList<String> cards;

    public Deck(ArrayList<String> cards) {
        this.cards = cards;
    }

    public ArrayList<String> getCards() {
        return cards;
    }

}
