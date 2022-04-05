package RecommenderSystem;

import com.megacrit.cardcrawl.cards.AbstractCard;
import com.megacrit.cardcrawl.relics.AbstractRelic;

import java.util.ArrayList;

public class StateBuilder {

    private ArrayList<String> deck = new ArrayList<>();
    private ArrayList<String> relics = new ArrayList<>();
    private ArrayList<String> choices = new ArrayList<>();
    private int floor;

    public StateBuilder setDeck(ArrayList<AbstractCard> deck) {
        for (AbstractCard toAdd : deck)
            this.deck.add(toAdd.cardID);
        return this;
    }

    public StateBuilder setRelics(ArrayList<AbstractRelic> relics) {
        for (AbstractRelic toAdd : relics)
            this.relics.add(toAdd.relicId);
        return this;
    }

    public StateBuilder setChoices(ArrayList<AbstractCard> choices) {
        for (AbstractCard toAdd : choices)
            this.choices.add(toAdd.cardID);
        return this;
    }

    public StateBuilder setFloor(int floor) {
        this.floor = floor;
        return this;
    }

    public State build() {
        State state = new State();
        state.setDeck(deck);
        state.setRelics(relics);
        state.setChoices(choices);
        state.setFloor(floor);
        deck = new ArrayList<>();
        relics = new ArrayList<>();
        choices = new ArrayList<>();
        return state;
    }

}
